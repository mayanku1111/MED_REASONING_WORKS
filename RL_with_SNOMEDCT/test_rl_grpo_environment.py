from rl_grpo_environment import GRPOEnvironment, GRPOPolicy, RewardBreakdown, rerank_evaluations_with_policy


def _sample_evaluations():
    return [
        {
            "concept_id": "111",
            "concept_name": "Pneumonia (disorder)",
            "score": 72,
            "confidence": 0.8,
            "reasoning_steps": ["Fever and productive cough support pneumonia"],
            "key_supporting_features": ["fever", "productive cough"],
            "contradicting_features": [],
        },
        {
            "concept_id": "222",
            "concept_name": "Pulmonary embolism (disorder)",
            "score": 69,
            "confidence": 0.7,
            "reasoning_steps": ["Sudden dyspnea and pleuritic pain can suggest PE"],
            "key_supporting_features": ["dyspnea", "pleuritic pain"],
            "contradicting_features": ["fever suggests infection instead"],
        },
    ]


def test_reward_shape():
    env = GRPOEnvironment()
    reward = env.compute_reward(
        evaluation=_sample_evaluations()[0],
        gold_concept_id="111",
        rank_index=0,
        evidence_traces=[{"name": "fever", "focus": "symptom"}, {"name": "productive cough", "focus": "symptom"}],
    )
    assert isinstance(reward, RewardBreakdown)
    assert reward.total <= 2.0
    assert reward.total >= -2.0


def test_rerank_returns_sorted_list():
    policy = GRPOPolicy()
    reranked = rerank_evaluations_with_policy(
        evaluations=_sample_evaluations(),
        evidence_traces=[{"name": "fever", "focus": "symptom"}, {"name": "productive cough", "focus": "symptom"}],
        policy=policy,
    )
    assert len(reranked) == 2
    assert reranked[0]["score"] >= reranked[1]["score"]


if __name__ == "__main__":
    test_reward_shape()
    test_rerank_returns_sorted_list()
    print("test_rl_grpo_environment.py passed")
