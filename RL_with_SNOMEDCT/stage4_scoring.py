from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
import json
import importlib
import os
import re

OpenRouter = None
try:
    OpenRouter = importlib.import_module("openrouter").OpenRouter
except Exception:
    OpenRouter = None

@dataclass
class ClinicalReasoningOutput:
    """Structured clinical reasoning from LLM"""
    organized_findings: Dict[str, List[str]]  # Categories: symptoms, signs, labs, etc.
    differential_reasoning: List[str]  # Step-by-step reasoning
    concept_scores: List[Dict]  # [{concept_id, score, reasoning}]
    confidence: float
    key_discriminating_features: List[str]

# ===== TWO-STEP CLINICAL REASONING PROMPTS =====

STEP1_ORGANIZE_SYSTEM = """You are a clinical reasoning expert organizing patient information.

Your task: Systematically organize clinical information into structured categories.

This mimics how physicians first gather and categorize information before reasoning.

Categories to use:
- Patient Demographics: age, sex, relevant background
- Chief Complaint: primary presenting problem
- History of Present Illness: timeline, associated symptoms, aggravating/relieving factors
- Past Medical History: relevant prior conditions
- Medications: current and relevant past medications
- Physical Examination: vital signs, examination findings
- Laboratory Data: lab values with units
- Imaging Findings: radiology results
- Negated Findings: explicitly absent symptoms/signs (IMPORTANT for ruling out)

Output ONLY valid JSON with these exact categories."""

STEP1_ORGANIZE_USER = """Organize this clinical information systematically.

CLINICAL PRESENTATION:
{clinical_text}

EXTRACTED EVIDENCE:
{evidence_traces}

Output JSON format:
{{
  "patient_demographics": ["55-year-old male"],
  "chief_complaint": ["chest pain"],
  "history_present_illness": ["crushing chest pain for 2 days", "pain radiates to left arm", "worse with exertion"],
  "past_medical_history": ["hypertension", "smoking"],
  "medications": [],
  "physical_examination": [],
  "laboratory_data": [],
  "imaging_findings": [],
  "negated_findings": ["no fever", "no cough"]
}}"""

STEP2_DIAGNOSE_SYSTEM = """You are a diagnostic reasoning expert evaluating SNOMED concepts.

Given ORGANIZED clinical information and candidate SNOMED concepts, use SYSTEMATIC clinical reasoning to score each concept.

Clinical Reasoning Framework:
1. PATTERN RECOGNITION: Does the concept match the clinical pattern?
2. DISCRIMINATING FEATURES: Are key diagnostic features present/absent?
3. NEGATION ANALYSIS: Do negated findings rule out this concept?
4. TEMPORAL CONSISTENCY: Does timeline fit this condition?
5. SEVERITY ALIGNMENT: Does severity match expectations?

For each concept, provide:
- Score (0-100): How well does it explain ALL the findings?
- Reasoning: Step-by-step clinical logic
- Discriminating features: What confirms or rules out this diagnosis?
- Confidence: How certain are you? (0.0-1.0)

CRITICAL RULES:
- If a negated finding contradicts the concept → Score MUST be < 20
- Consider ALL categories, not just symptoms
- Explain your reasoning like teaching a medical student

Output ONLY valid JSON."""

STEP2_DIAGNOSE_USER = """Score these SNOMED concepts using clinical reasoning.

=== ORGANIZED CLINICAL INFORMATION ===
{organized_clinical_info}

=== CANDIDATE CONCEPTS TO EVALUATE ===
{candidate_concepts}

For EACH concept, provide reasoning and score.

Output JSON format:
{{
  "concept_evaluations": [
    {{
      "concept_id": "22298006",
      "concept_name": "Myocardial infarction (disorder)",
      "score": 85,
      "reasoning_steps": [
        "Pattern Recognition: Classic presentation of acute MI - crushing chest pain with radiation",
        "Discriminating Features: Substernal location, radiation to arm, diaphoresis all support MI",
        "Risk Factors: Hypertension and smoking are major MI risk factors",
        "Negation Analysis: Absence of fever/cough doesn't contradict MI",
        "Temporal: 2-day onset fits ACS timeline"
      ],
      "key_supporting_features": ["crushing chest pain", "radiation to arm", "exertional", "risk factors"],
      "contradicting_features": [],
      "confidence": 0.9,
      "clinical_urgency": "high"
    }},
    {{
      "concept_id": "29857009",
      "concept_name": "Chest pain (finding)",
      "score": 40,
      "reasoning_steps": [
        "Pattern Recognition: Too generic - doesn't explain the SPECIFIC presentation",
        "Discriminating Features: Fails to account for radiation, exertional nature, risk factors",
        "Clinical Assessment: This is a symptom, not a diagnosis - lacks explanatory power"
      ],
      "key_supporting_features": ["chest pain present"],
      "contradicting_features": ["too non-specific"],
      "confidence": 0.6,
      "clinical_urgency": "medium"
    }}
  ],
  "differential_summary": "Top diagnosis is MI based on classic presentation with key discriminating features. Chest pain alone is too non-specific.",
  "overall_confidence": 0.85,
  "recommended_next_steps": ["ECG", "troponin", "cardiology consult"]
}}"""

# ===== REASONING-ENHANCED SCORER =====

@dataclass
class PipelineScoredConcept:
    """Compatibility object expected by pipeline_main.py (has .total_score)."""
    concept_id: str
    fsn: str
    total_score: float
    confidence: float = 0.0

    # Common legacy fields some pipelines expect
    attribute_scores: Dict[str, float] = field(default_factory=dict)
    reasoning_steps: List[str] = field(default_factory=list)
    key_supporting_features: List[str] = field(default_factory=list)
    contradicting_features: List[str] = field(default_factory=list)

    # Fields consumed by pipeline_main.py / stage6 / stage8
    supporting_traces: List[str] = field(default_factory=list)
    contradicting_traces: List[str] = field(default_factory=list)
    score_breakdown: Dict[str, float] = field(default_factory=dict)
    llm_reasoning: str = ""

    # Keep access to original data
    raw: Dict[str, Any] = field(default_factory=dict)
    concept: Optional[Any] = None

class ClinicalReasoningScorer:
    """LLM-first scorer using structured clinical reasoning"""
    
    def __init__(
        self,
        api_key: str,
        model: str = "google/gemini-2.5-flash-lite",  # Best for reasoning
        fallback_model: str = "google/gemini-2.5-flash-lite",
        use_llm_scoring: bool = True,
        use_rl_reranker: bool = False,
        rl_policy_path: Optional[str] = None,
        rl_reward_weights: Optional[Dict[str, float]] = None,
        **_kwargs,  # tolerate unexpected/legacy kwargs from the pipeline
    ):
        self.api_key = api_key
        self.model = model
        self.fallback_model = fallback_model
        self.use_llm_scoring = use_llm_scoring
        self.reasoning_cache = {}
        self.use_rl_reranker = use_rl_reranker
        self.rl_policy_path = rl_policy_path or os.getenv("RL_POLICY_PATH") or "rl_policy.json"
        self.rl_reward_weights = rl_reward_weights or {}
        self.rl_policy = None

        if self.use_rl_reranker:
            try:
                from rl_grpo_environment import load_policy

                self.rl_policy = load_policy(self.rl_policy_path)
                print(f"🤖 RL reranker enabled: loaded policy from {self.rl_policy_path}")
            except Exception as e:
                print(f"⚠️ RL policy unavailable, continuing without RL reranking: {str(e)[:80]}")
                self.use_rl_reranker = False
    
    
    def _call_llm(self, system: str, user: str, model: str = None) -> dict:
        """Call LLM with error handling"""
        if not getattr(self, "use_llm_scoring", True):
            return None
        if OpenRouter is None:
            print("⚠️ OpenRouter package not available; skipping LLM scoring")
            return None
        if model is None:
            model = self.model
            
        try:
            with OpenRouter(api_key=self.api_key) as client:
                response = client.chat.send(
                    model=model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.3,  # Balance creativity vs consistency
                    reasoning={"enabled": False}
                )
                
                content = response.choices[0].message.content
                
                # Extract JSON
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
                return json.loads(content)
                
        except Exception as e:
            print(f"⚠️ LLM error with {model}: {str(e)[:80]}")
            # Try fallback
            if model != self.fallback_model:
                print(f"   🔄 Trying fallback: {self.fallback_model}")
                return self._call_llm(system, user, model=self.fallback_model)
            return None

    def step1_organize_clinical_info(self, clinical_text: str, evidence_traces: List) -> Dict:
        """
        Step 1: Organize clinical information into structured categories
        This mimics physician's initial information gathering
        """
        print("🏥 Step 1: Organizing clinical information...")

        # Format evidence traces (accept Trace objects, dicts, or strings)
        evidence_str = ""
        for trace in (evidence_traces or []):
            if isinstance(trace, str):
                name = trace
                focus = "evidence"
                attrs_dict = {}
            elif isinstance(trace, dict):
                name = trace.get("name") or trace.get("text") or trace.get("value") or str(trace)
                focus = trace.get("focus") or trace.get("type") or "evidence"
                attrs_dict = trace.get("attributes") or {}
            else:
                name = getattr(trace, "name", str(trace))
                focus = getattr(trace, "focus", "evidence")
                attrs_dict = getattr(trace, "attributes", {}) or {}

            # attrs_dict might not be a dict
            if not isinstance(attrs_dict, dict):
                attrs_dict = {}

            attrs = ", ".join([f"{k}={v}" for k, v in attrs_dict.items() if v])
            evidence_str += f"- {name} [{focus}]"
            if attrs:
                evidence_str += f" ({attrs})"
            if str(focus).lower() == "negation":
                evidence_str += " [NEGATED]"
            evidence_str += "\n"

        result = self._call_llm(
            STEP1_ORGANIZE_SYSTEM,
            STEP1_ORGANIZE_USER.format(
                clinical_text=clinical_text,
                evidence_traces=evidence_str
            )
        )

        if result:
            print("   ✓ Clinical information organized")
            return result

        print("   ⚠️ Using rule-based fallback")
        return self._fallback_organize(evidence_traces)

    def _fallback_organize(self, evidence_traces: List) -> Dict:
        """Rule-based fallback if LLM fails (accept Trace objects, dicts, or strings)"""
        organized = {
            "patient_demographics": [],
            "chief_complaint": [],
            "history_present_illness": [],
            "past_medical_history": [],
            "medications": [],
            "physical_examination": [],
            "laboratory_data": [],
            "imaging_findings": [],
            "negated_findings": []
        }

        for trace in (evidence_traces or []):
            if isinstance(trace, str):
                name = trace
                focus = "symptom"  # best-effort default
            elif isinstance(trace, dict):
                name = trace.get("name") or trace.get("text") or trace.get("value") or str(trace)
                focus = trace.get("focus") or trace.get("type") or "symptom"
            else:
                name = getattr(trace, "name", str(trace))
                focus = getattr(trace, "focus", "symptom")

            focus = str(focus).lower()

            if focus == "negation":
                organized["negated_findings"].append(f"no {name}")
            elif focus == "symptom":
                organized["history_present_illness"].append(name)
            elif focus == "sign":
                organized["physical_examination"].append(name)
            elif focus == "lab":
                organized["laboratory_data"].append(name)
            elif focus == "imaging":
                organized["imaging_findings"].append(name)
            elif focus == "history":
                organized["past_medical_history"].append(name)
            else:
                organized["history_present_illness"].append(name)

        return organized
    
    def step2_score_concepts_with_reasoning(
        self,
        organized_info: Dict,
        candidate_concepts: List,
        top_k: int = 10
    ) -> List[Dict]:
        """
        Step 2: Score concepts using structured clinical reasoning
        This is where the REAL diagnostic thinking happens
        """
        
        print(f"🧠 Step 2: Clinical reasoning for {len(candidate_concepts)} candidates...")
        
        # Format organized info
        organized_str = json.dumps(organized_info, indent=2)
        
        # Format candidates (limit to top_k for efficiency)
        candidates_to_score = candidate_concepts[:top_k]
        candidates_str = ""
        for i, concept in enumerate(candidates_to_score, 1):
            candidates_str += f"{i}. ID: {concept.concept_id}\n"
            candidates_str += f"   Name: {concept.fsn}\n"
            if hasattr(concept, 'matched_term'):
                candidates_str += f"   Matched via: {concept.matched_term}\n"
            candidates_str += "\n"
        
        result = self._call_llm(
            STEP2_DIAGNOSE_SYSTEM,
            STEP2_DIAGNOSE_USER.format(
                organized_clinical_info=organized_str,
                candidate_concepts=candidates_str
            )
        )
        
        if not result or "concept_evaluations" not in result:
            print("   ⚠️ LLM reasoning failed, using rule-based scoring")
            return self._fallback_scoring(candidate_concepts, organized_info)
        
        evaluations = result["concept_evaluations"]
        
        print(f"   ✓ Reasoned through {len(evaluations)} concepts")
        if evaluations:
            top = evaluations[0]
            print(f"   🏆 Top: {top['concept_name']} (score: {top['score']})")
            print(f"      Reasoning: {top['reasoning_steps'][0]}")
        
        return evaluations
    
    def _fallback_scoring(self, candidates: List, organized_info: Dict) -> List[Dict]:
        """Simple rule-based fallback"""
        evaluations = []
        
        negated = organized_info.get("negated_findings", [])
        
        for concept in candidates[:10]:
            # Check for negation conflicts
            score = 50  # Base score
            contradicting = []
            
            for neg in negated:
                neg_term = neg.replace("no ", "").lower()
                if neg_term in concept.fsn.lower():
                    score = 15
                    contradicting.append(neg)
            
            evaluations.append({
                "concept_id": concept.concept_id,
                "concept_name": concept.fsn,
                "score": score,
                "reasoning_steps": ["Rule-based fallback scoring"],
                "key_supporting_features": [],
                "contradicting_features": contradicting,
                "confidence": 0.5
            })
        
        evaluations.sort(key=lambda x: x['score'], reverse=True)
        return evaluations

    def _extract_evidence_terms(self, evidence_traces: List) -> List[str]:
        terms: List[str] = []
        for trace in (evidence_traces or []):
            if isinstance(trace, str):
                terms.append(trace)
                continue

            if isinstance(trace, dict):
                name = trace.get("name") or trace.get("text") or trace.get("value")
                if name:
                    terms.append(str(name))
                attrs = trace.get("attributes") or {}
                if isinstance(attrs, dict):
                    terms.extend([str(v) for v in attrs.values() if v])
                continue

            name = getattr(trace, "name", None)
            if name:
                terms.append(str(name))
            attrs = getattr(trace, "attributes", {}) or {}
            if isinstance(attrs, dict):
                terms.extend([str(v) for v in attrs.values() if v])

        return terms

    def _convert_evals_to_pipeline_objects(
        self,
        evaluations: List[Dict],
        concept_by_id: Dict[str, Any],
        evidence_terms: List[str],
    ) -> List[PipelineScoredConcept]:
        converted: List[PipelineScoredConcept] = []

        for ev in (evaluations or []):
            concept_id = str(ev.get("concept_id", ""))
            concept_obj = concept_by_id.get(concept_id)

            fsn = ev.get("concept_name")
            if not fsn and concept_obj is not None:
                fsn = getattr(concept_obj, "fsn", "")
            if not fsn:
                fsn = "Unknown concept"

            score = float(ev.get("score", 0.0) or 0.0)
            confidence = float(ev.get("confidence", 0.0) or 0.0)
            reasoning_steps = ev.get("reasoning_steps") or []
            if not isinstance(reasoning_steps, list):
                reasoning_steps = [str(reasoning_steps)]

            supporting = ev.get("key_supporting_features") or []
            contradicting = ev.get("contradicting_features") or []

            if not supporting:
                supporting = [t for t in evidence_terms[:5] if t]

            score_breakdown = {
                "clinical_reasoning_score": score,
                "confidence": confidence,
                "support_match_count": float(len(supporting)),
                "contradiction_count": float(len(contradicting)),
            }

            llm_reasoning = " | ".join([str(s) for s in reasoning_steps[:3] if s])

            converted.append(
                PipelineScoredConcept(
                    concept_id=concept_id,
                    fsn=str(fsn),
                    total_score=score,
                    confidence=confidence,
                    reasoning_steps=[str(s) for s in reasoning_steps],
                    key_supporting_features=[str(s) for s in supporting],
                    contradicting_features=[str(s) for s in contradicting],
                    supporting_traces=[str(s) for s in supporting],
                    contradicting_traces=[str(s) for s in contradicting],
                    score_breakdown=score_breakdown,
                    llm_reasoning=llm_reasoning,
                    raw=ev,
                    concept=concept_obj,
                )
            )

        converted.sort(key=lambda x: x.total_score, reverse=True)
        return converted

    def _flatten_candidates(self, snomed_candidates: Dict) -> Tuple[List[Any], Dict[str, Any]]:
        all_candidates: List[Any] = []
        concept_by_id: Dict[str, Any] = {}
        seen: set = set()

        for _, candidates in (snomed_candidates or {}).items():
            for concept in (candidates or []):
                cid = getattr(concept, "concept_id", None)
                if not cid:
                    continue
                cid = str(cid)
                concept_by_id[cid] = concept
                if cid not in seen:
                    seen.add(cid)
                    all_candidates.append(concept)

        return all_candidates, concept_by_id

    def _apply_rl_reranker(self, evaluations: List[Dict], evidence_traces: List) -> List[Dict]:
        if not self.use_rl_reranker or not self.rl_policy or not evaluations:
            return evaluations

        try:
            from rl_grpo_environment import rerank_evaluations_with_policy

            reranked = rerank_evaluations_with_policy(
                evaluations=evaluations,
                evidence_traces=evidence_traces,
                policy=self.rl_policy,
            )
            if reranked:
                return reranked
        except Exception as e:
            print(f"⚠️ RL reranking failed, using baseline ranking: {str(e)[:80]}")

        return evaluations
    
    def aggregate_scores(self, *args, **kwargs):
        """
        Backward-compatible entrypoint expected by pipeline_main.py.
        Returns a list of PipelineScoredConcept objects.
        """
        clinical_text = kwargs.get("clinical_text") or kwargs.get("query") or kwargs.get("text") or ""
        evidence_traces = kwargs.get("evidence_traces") or kwargs.get("evidence") or []
        hypotheses = kwargs.get("hypotheses") or []
        snomed_candidates = kwargs.get("snomed_candidates")
        candidate_concepts = kwargs.get("candidate_concepts") or kwargs.get("candidates")
        top_k = kwargs.get("top_k", 15)

        if args:
            # Common pipeline call pattern:
            # aggregate_scores(snomed_candidates, evidence_traces, hypotheses, counterfactuals)
            if isinstance(args[0], dict):
                snomed_candidates = args[0]
                if len(args) > 1:
                    evidence_traces = args[1]
                if len(args) > 2:
                    hypotheses = args[2]
            elif isinstance(args[0], str):
                # Alternate pattern: (clinical_text, evidence_traces, hypotheses, snomed_candidates)
                clinical_text = args[0]
                if len(args) > 1:
                    evidence_traces = args[1]
                if len(args) > 2:
                    hypotheses = args[2]
                if len(args) > 3 and isinstance(args[3], dict):
                    snomed_candidates = args[3]
            elif len(args) > 0:
                # Fallback positional parsing
                evidence_traces = args[0]
                if len(args) > 1:
                    hypotheses = args[1]
                if len(args) > 2 and isinstance(args[2], dict):
                    snomed_candidates = args[2]

        if not clinical_text:
            clinical_text = " ".join(self._extract_evidence_terms(evidence_traces)[:20])

        concept_by_id: Dict[str, Any] = {}

        if isinstance(snomed_candidates, dict):
            all_candidates, concept_by_id = self._flatten_candidates(snomed_candidates)
            organized_info = self.step1_organize_clinical_info(clinical_text, evidence_traces)
            evals = self.step2_score_concepts_with_reasoning(
                organized_info=organized_info,
                candidate_concepts=all_candidates,
                top_k=top_k,
            )
        else:
            if candidate_concepts is None:
                candidate_concepts = []

            for concept in candidate_concepts:
                cid = getattr(concept, "concept_id", None)
                if cid:
                    concept_by_id[str(cid)] = concept

            organized_info = self.step1_organize_clinical_info(clinical_text, evidence_traces)
            evals = self.step2_score_concepts_with_reasoning(
                organized_info=organized_info,
                candidate_concepts=candidate_concepts,
                top_k=top_k,
            )

        evals = self._apply_rl_reranker(evals, evidence_traces)
        evidence_terms = self._extract_evidence_terms(evidence_traces)
        return self._convert_evals_to_pipeline_objects(evals, concept_by_id, evidence_terms)

    def score_all_candidates(
        self,
        clinical_text: str,
        evidence_traces: List,
        hypotheses: List,
        snomed_candidates: Dict
    ) -> Dict:
        """
        Complete two-step reasoning pipeline
        
        Returns:
            Dict with 'organized_info', 'concept_scores', 'reasoning'
        """
        
        # Step 1: Organize
        organized_info = self.step1_organize_clinical_info(clinical_text, evidence_traces)
        
        # Collect all unique candidates
        all_candidates = []
        seen_ids = set()
        
        for trace_key, candidates in snomed_candidates.items():
            for concept in candidates:
                if concept.concept_id not in seen_ids:
                    all_candidates.append(concept)
                    seen_ids.add(concept.concept_id)
        
        print(f"   📊 Total unique candidates: {len(all_candidates)}")
        
        # Step 2: Reason and score
        concept_evaluations = self.step2_score_concepts_with_reasoning(
            organized_info,
            all_candidates,
            top_k=15  # Score top 15 candidates
        )
        
        return {
            "organized_info": organized_info,
            "concept_evaluations": concept_evaluations,
            "total_candidates_considered": len(all_candidates)
        }


# ===== INTEGRATION WITH YOUR PIPELINE =====

def integrate_reasoning_scorer():
    """Example of how to use this in your pipeline"""
    from stage1_traces import MultiAgentTraceGenerator
    from stage3_retrieval import SNOMEDRetriever
    import os
    
    # Your existing pipeline
    trace_gen = MultiAgentTraceGenerator(api_key=os.getenv("OPENROUTER_API_KEY"))
    retriever = SNOMEDRetriever(api_key=os.getenv("OPENROUTER_API_KEY"))
    
    # NEW: Reasoning scorer (replaces Stage 4-5)
    reasoning_scorer = ClinicalReasoningScorer(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        model="google/gemini-2.5-flash-lite"  # Best for clinical reasoning
    )
    
    # Test case
    clinical_text = """
    55-year-old male with history of hypertension and smoking presents with 
    crushing chest pain for 2 days. Pain radiates to left arm and jaw, 
    worse with exertion. Associated with diaphoresis. No fever. No cough.
    """
    
    # Stage 1: Extract traces
    traces = trace_gen.generate_all_traces(clinical_text, [])
    
    # Stage 3: Retrieve candidates
    candidates = retriever.retrieve_for_traces(
        traces["evidence"],
        traces["hypotheses"]
    )
    
    # Stage 4-5: LLM-first reasoning (REPLACES your current scoring)
    results = reasoning_scorer.score_all_candidates(
        clinical_text,
        traces["evidence"],
        traces["hypotheses"],
        candidates
    )
    
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    
    print("\n📋 Organized Clinical Information:")
    for category, items in results["organized_info"].items():
        if items:
            print(f"  {category}: {items}")
    
    print("\n🎯 Top 3 Diagnoses with Reasoning:")
    for i, eval in enumerate(results["concept_evaluations"][:3], 1):
        print(f"\n{i}. {eval['concept_name']}")
        print(f"   Score: {eval['score']}/100")
        print(f"   Confidence: {eval['confidence']:.2f}")
        print(f"   Reasoning:")
        for step in eval['reasoning_steps'][:3]:
            print(f"      • {step}")
        
        if eval['contradicting_features']:
            print(f"   ⚠️  Contradictions: {eval['contradicting_features']}")

if __name__ == "__main__":
    integrate_reasoning_scorer()
EnhancedAttributeScorer = ClinicalReasoningScorer