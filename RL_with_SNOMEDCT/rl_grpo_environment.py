from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional
import json
import math
import random
import re


DEFAULT_REWARD_WEIGHTS = {
    "final_answer": 0.55,
    "reasoning_alignment": 0.30,
    "negation_consistency": 0.10,
    "calibration": 0.05,
}

DEFAULT_FEATURE_WEIGHTS = {
    "base_score": 0.25,
    "confidence": 0.15,
    "alignment": 0.45,
    "negation_safe": 0.15,
}


@dataclass
class GRPOPolicy:
    feature_weights: Dict[str, float] = field(default_factory=lambda: DEFAULT_FEATURE_WEIGHTS.copy())
    reward_weights: Dict[str, float] = field(default_factory=lambda: DEFAULT_REWARD_WEIGHTS.copy())
    temperature: float = 1.0
    version: str = "1.0"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GRPOPolicy":
        return cls(
            feature_weights=dict(data.get("feature_weights") or DEFAULT_FEATURE_WEIGHTS),
            reward_weights=dict(data.get("reward_weights") or DEFAULT_REWARD_WEIGHTS),
            temperature=float(data.get("temperature", 1.0)),
            version=str(data.get("version", "1.0")),
        )


@dataclass
class RewardBreakdown:
    final_answer: float
    reasoning_alignment: float
    negation_consistency: float
    calibration: float
    total: float


@dataclass
class GRPOCase:
    clinical_text: str
    gold_concept_id: str
    evidence_traces: List[Any] = field(default_factory=list)
    candidate_concepts: List[Any] = field(default_factory=list)
    case_id: str = ""


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _trace_name(trace: Any) -> str:
    if isinstance(trace, str):
        return trace
    if isinstance(trace, dict):
        return str(trace.get("name") or trace.get("text") or trace.get("value") or "")
    return str(getattr(trace, "name", ""))


def _trace_focus(trace: Any) -> str:
    if isinstance(trace, dict):
        return str(trace.get("focus") or trace.get("type") or "")
    return str(getattr(trace, "focus", ""))


def _trace_attributes(trace: Any) -> Dict[str, Any]:
    if isinstance(trace, dict):
        attrs = trace.get("attributes") or {}
    else:
        attrs = getattr(trace, "attributes", {}) or {}
    if not isinstance(attrs, dict):
        return {}
    return attrs


def extract_evidence_terms(evidence_traces: List[Any]) -> List[str]:
    terms: List[str] = []
    for trace in evidence_traces or []:
        name = _trace_name(trace)
        if name:
            terms.append(name)
        attrs = _trace_attributes(trace)
        for value in attrs.values():
            if value:
                terms.append(str(value))
    return terms


def extract_negated_terms(evidence_traces: List[Any]) -> List[str]:
    negated: List[str] = []
    for trace in evidence_traces or []:
        focus = _trace_focus(trace).lower()
        if focus == "negation":
            name = _trace_name(trace)
            if name:
                negated.append(name)
    return negated


def _reasoning_text(evaluation: Dict[str, Any]) -> str:
    steps = evaluation.get("reasoning_steps") or []
    if not isinstance(steps, list):
        steps = [str(steps)]
    supporting = evaluation.get("key_supporting_features") or []
    contradicting = evaluation.get("contradicting_features") or []
    chunks = [" ".join([str(step) for step in steps]), " ".join([str(v) for v in supporting]), " ".join([str(v) for v in contradicting])]
    return " ".join([c for c in chunks if c]).strip()


def _concept_text(evaluation: Dict[str, Any]) -> str:
    parts = [
        str(evaluation.get("concept_name", "")),
        str(evaluation.get("concept_id", "")),
    ]
    return " ".join(parts).strip()


def _overlap_score(a_terms: List[str], b_terms: List[str]) -> float:
    a = set([t for t in a_terms if t])
    b = set([t for t in b_terms if t])
    if not a or not b:
        return 0.0
    inter = len(a.intersection(b))
    union = len(a.union(b))
    return float(inter) / float(union)


class GRPOEnvironment:
    """Lightweight GRPO-style reward environment for SNOMED reasoning tasks."""

    def __init__(
        self,
        policy: Optional[GRPOPolicy] = None,
        reward_weights: Optional[Dict[str, float]] = None,
        seed: int = 13,
    ):
        self.policy = policy or GRPOPolicy()
        if reward_weights:
            self.policy.reward_weights.update(reward_weights)
        self.rng = random.Random(seed)

    def reset(self, case: GRPOCase) -> GRPOCase:
        return case

    def sample_group(self, evaluations: List[Dict[str, Any]], group_size: int) -> List[Dict[str, Any]]:
        if not evaluations:
            return []
        group_size = max(1, min(group_size, len(evaluations)))
        if group_size == len(evaluations):
            sampled = [dict(ev) for ev in evaluations]
            self.rng.shuffle(sampled)
            return sampled
        return [dict(ev) for ev in self.rng.sample(evaluations, group_size)]

    def build_features(
        self,
        evaluation: Dict[str, Any],
        evidence_terms: List[str],
        negated_terms: List[str],
    ) -> Dict[str, float]:
        base_score = float(evaluation.get("score", 0.0) or 0.0) / 100.0
        confidence = float(evaluation.get("confidence", 0.0) or 0.0)
        reasoning_tokens = _tokenize(_reasoning_text(evaluation))
        concept_tokens = _tokenize(_concept_text(evaluation))
        evidence_tokens = _tokenize(" ".join(evidence_terms))
        neg_tokens = _tokenize(" ".join(negated_terms))

        alignment = 0.5 * _overlap_score(reasoning_tokens, evidence_tokens) + 0.5 * _overlap_score(reasoning_tokens, concept_tokens)

        contradiction = 0.0
        if neg_tokens:
            if set(neg_tokens).intersection(set(reasoning_tokens)):
                contradiction = 1.0
            if set(neg_tokens).intersection(set(concept_tokens)):
                contradiction = 1.0

        return {
            "base_score": max(0.0, min(1.0, base_score)),
            "confidence": max(0.0, min(1.0, confidence)),
            "alignment": max(0.0, min(1.0, alignment)),
            "negation_safe": 1.0 - contradiction,
        }

    def score_with_policy(self, evaluation: Dict[str, Any], evidence_traces: List[Any]) -> float:
        evidence_terms = extract_evidence_terms(evidence_traces)
        negated_terms = extract_negated_terms(evidence_traces)
        features = self.build_features(evaluation, evidence_terms, negated_terms)

        linear = 0.0
        for feature_name, feature_value in features.items():
            linear += self.policy.feature_weights.get(feature_name, 0.0) * feature_value

        temperature = self.policy.temperature if self.policy.temperature > 1e-6 else 1.0
        return _sigmoid(linear / temperature)

    def compute_reward(
        self,
        evaluation: Dict[str, Any],
        gold_concept_id: str,
        rank_index: int,
        evidence_traces: List[Any],
    ) -> RewardBreakdown:
        predicted = str(evaluation.get("concept_id", ""))
        correct = predicted == str(gold_concept_id)

        final_answer = 1.0 if correct else (1.0 / float(rank_index + 2))

        evidence_terms = extract_evidence_terms(evidence_traces)
        negated_terms = extract_negated_terms(evidence_traces)
        features = self.build_features(evaluation, evidence_terms, negated_terms)

        reasoning_alignment = features["alignment"]
        negation_consistency = features["negation_safe"] if correct else (features["negation_safe"] - 1.0)

        confidence = float(evaluation.get("confidence", 0.0) or 0.0)
        calibration = confidence if correct else (-0.5 * confidence)

        weights = self.policy.reward_weights
        total = (
            weights.get("final_answer", DEFAULT_REWARD_WEIGHTS["final_answer"]) * final_answer
            + weights.get("reasoning_alignment", DEFAULT_REWARD_WEIGHTS["reasoning_alignment"]) * reasoning_alignment
            + weights.get("negation_consistency", DEFAULT_REWARD_WEIGHTS["negation_consistency"]) * negation_consistency
            + weights.get("calibration", DEFAULT_REWARD_WEIGHTS["calibration"]) * calibration
        )

        return RewardBreakdown(
            final_answer=final_answer,
            reasoning_alignment=reasoning_alignment,
            negation_consistency=negation_consistency,
            calibration=calibration,
            total=total,
        )

    def compute_group_advantages(self, rewards: List[float]) -> List[float]:
        if not rewards:
            return []
        mean = sum(rewards) / len(rewards)
        variance = sum((r - mean) ** 2 for r in rewards) / len(rewards)
        std = math.sqrt(variance) if variance > 1e-8 else 1.0
        return [(r - mean) / std for r in rewards]

    def rerank_evaluations(
        self,
        evaluations: List[Dict[str, Any]],
        evidence_traces: List[Any],
    ) -> List[Dict[str, Any]]:
        reranked: List[Dict[str, Any]] = []

        for evaluation in evaluations or []:
            ev = dict(evaluation)
            base = float(ev.get("score", 0.0) or 0.0) / 100.0
            policy_score = self.score_with_policy(ev, evidence_traces)
            blended = (0.65 * base) + (0.35 * policy_score)

            ev["rl_base_score"] = base
            ev["rl_policy_score"] = policy_score
            ev["score"] = max(0.0, min(100.0, blended * 100.0))
            reranked.append(ev)

        reranked.sort(key=lambda x: float(x.get("score", 0.0) or 0.0), reverse=True)
        return reranked


def rerank_evaluations_with_policy(
    evaluations: List[Dict[str, Any]],
    evidence_traces: List[Any],
    policy: GRPOPolicy,
) -> List[Dict[str, Any]]:
    env = GRPOEnvironment(policy=policy)
    return env.rerank_evaluations(evaluations, evidence_traces)


class GRPOTrainer:
    """Offline trainer using grouped relative rewards (GRPO-style)."""

    def __init__(
        self,
        environment: Optional[GRPOEnvironment] = None,
        learning_rate: float = 0.05,
        group_size: int = 4,
        seed: int = 13,
    ):
        self.env = environment or GRPOEnvironment(seed=seed)
        self.learning_rate = learning_rate
        self.group_size = group_size
        self.rng = random.Random(seed)

    def _load_cases_from_jsonl(self, dataset_path: str) -> List[GRPOCase]:
        cases: List[GRPOCase] = []
        with open(dataset_path, "r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                raw = line.strip()
                if not raw:
                    continue
                try:
                    row = json.loads(raw)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSONL at line {line_no}: {e}") from e

                gold = str(row.get("gold_concept_id") or "")
                if not gold:
                    continue

                cases.append(
                    GRPOCase(
                        case_id=str(row.get("case_id") or f"case_{line_no}"),
                        clinical_text=str(row.get("clinical_text") or ""),
                        gold_concept_id=gold,
                        evidence_traces=list(row.get("evidence_traces") or []),
                        candidate_concepts=list(row.get("candidate_concepts") or []),
                    )
                )
        return cases

    def _update_policy_weights(
        self,
        sampled_evals: List[Dict[str, Any]],
        advantages: List[float],
        evidence_terms: List[str],
        negated_terms: List[str],
    ) -> None:
        for evaluation, advantage in zip(sampled_evals, advantages):
            features = self.env.build_features(evaluation, evidence_terms, negated_terms)
            for key, value in features.items():
                current = self.env.policy.feature_weights.get(key, 0.0)
                self.env.policy.feature_weights[key] = current + (self.learning_rate * advantage * value)

    def train_from_cases(
        self,
        cases: List[GRPOCase],
        scorer_fn: Callable[[GRPOCase], List[Dict[str, Any]]],
        epochs: int = 3,
    ) -> Dict[str, Any]:
        if not cases:
            return {
                "epochs": 0,
                "cases": 0,
                "mean_reward": 0.0,
                "top1_accuracy": 0.0,
                "feature_weights": dict(self.env.policy.feature_weights),
            }

        total_reward = 0.0
        total_steps = 0
        total_top1_correct = 0

        for _ in range(max(1, epochs)):
            self.rng.shuffle(cases)

            for case in cases:
                evaluations = scorer_fn(case) or []
                if not evaluations:
                    continue

                evaluations.sort(key=lambda x: float(x.get("score", 0.0) or 0.0), reverse=True)

                top1 = str(evaluations[0].get("concept_id", ""))
                if top1 == str(case.gold_concept_id):
                    total_top1_correct += 1

                sampled = self.env.sample_group(evaluations, self.group_size)
                evidence_terms = extract_evidence_terms(case.evidence_traces)
                negated_terms = extract_negated_terms(case.evidence_traces)

                rewards: List[float] = []
                for rank_idx, ev in enumerate(sampled):
                    reward = self.env.compute_reward(ev, case.gold_concept_id, rank_idx, case.evidence_traces)
                    rewards.append(reward.total)
                    total_reward += reward.total
                    total_steps += 1

                advantages = self.env.compute_group_advantages(rewards)
                self._update_policy_weights(sampled, advantages, evidence_terms, negated_terms)

        mean_reward = total_reward / float(total_steps) if total_steps else 0.0
        top1_accuracy = total_top1_correct / float(len(cases) * max(1, epochs))

        return {
            "epochs": max(1, epochs),
            "cases": len(cases),
            "mean_reward": mean_reward,
            "top1_accuracy": top1_accuracy,
            "feature_weights": dict(self.env.policy.feature_weights),
        }

    def train_from_jsonl(
        self,
        dataset_path: str,
        scorer_fn: Callable[[GRPOCase], List[Dict[str, Any]]],
        epochs: int = 3,
        output_policy_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        cases = self._load_cases_from_jsonl(dataset_path)
        summary = self.train_from_cases(cases, scorer_fn=scorer_fn, epochs=epochs)
        if output_policy_path:
            save_policy(output_policy_path, self.env.policy)
            summary["policy_path"] = output_policy_path
        return summary


def save_policy(path: str, policy: GRPOPolicy) -> None:
    payload = {
        "feature_weights": dict(policy.feature_weights),
        "reward_weights": dict(policy.reward_weights),
        "temperature": policy.temperature,
        "version": policy.version,
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_policy(path: str) -> GRPOPolicy:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return GRPOPolicy.from_dict(payload)


def policy_to_dict(policy: GRPOPolicy) -> Dict[str, Any]:
    return asdict(policy)
