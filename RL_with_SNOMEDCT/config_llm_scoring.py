"""
Configuration for LLM-Based Scoring Methods
Easily switch between different prompting techniques
"""

LLM_SCORING_CONFIG = {
    # Enable/disable LLM-based scoring
    "enabled": True,
    
    # Scoring method: "cot", "react", "few-shot", "self-consistency", "hybrid"
    "method": "cot",
    
    # How many top candidates to score with LLM (expensive!)
    "max_candidates_to_score": 5,
    
    # Self-consistency: how many samples to run
    "self_consistency_samples": 3,
    
    # Temperature for LLM scoring
    "temperature": 0.3,
    
    # Model for scoring (can be different from main model)
    "scoring_model": "google/gemini-2.5-flash-lite",
    
    # Weights for hybrid scoring
    "hybrid_weights": {
        "rule_based": 0.6,  # 60% weight to rule-based
        "llm_based": 0.4    # 40% weight to LLM
    }
}

# ==================== PROMPTING TECHNIQUE DESCRIPTIONS ====================

TECHNIQUE_INFO = {
    "cot": {
        "name": "Chain-of-Thought (CoT)",
        "description": "LLM reasons step-by-step through evidence",
        "pros": ["Clear reasoning", "Good for complex cases", "Interpretable"],
        "cons": ["Slower", "More tokens"],
        "best_for": "Complex multi-symptom presentations",
        "example": "Step 1: Check symptom X. Step 2: Evaluate temporal pattern. Step 3: Consider negations..."
    },
    
    "react": {
        "name": "ReAct (Reasoning + Acting)",
        "description": "Alternates between thinking and checking facts",
        "pros": ["Systematic", "Good for verification", "Structured"],
        "cons": ["Verbose", "Can be repetitive"],
        "best_for": "Cases needing careful verification",
        "example": "Thought: Need to check X. Action: Checking X. Observation: Found Y..."
    },
    
    "few-shot": {
        "name": "Few-Shot Learning",
        "description": "Learn from examples, then score new case",
        "pros": ["Fast", "Consistent", "Good calibration"],
        "cons": ["Limited by examples", "Less flexible"],
        "best_for": "Standard presentations similar to training examples",
        "example": "Example 1: MI with crushing pain = 9/10. Your case: Similar pattern = 8/10"
    },
    
    "self-consistency": {
        "name": "Self-Consistency",
        "description": "Run multiple times, take majority vote",
        "pros": ["More reliable", "Reduces errors", "Confidence measure"],
        "cons": ["Expensive (multiple calls)", "Slower"],
        "best_for": "High-stakes decisions, borderline cases",
        "example": "Run 5 times: [8, 9, 8, 9, 8] → Consensus: 8/10"
    },
    
    "hybrid": {
        "name": "Hybrid (Rules + LLM)",
        "description": "Combine rule-based and LLM scoring",
        "pros": ["Best of both worlds", "Balanced", "Efficient"],
        "cons": ["More complex", "Needs tuning"],
        "best_for": "Production systems",
        "example": "Rule score: 6.0, LLM score: 8.0 → Weighted: 6.8"
    }
}

# ==================== PERFORMANCE PROFILES ====================

PERFORMANCE_PROFILES = {
    "fast": {
        "enabled": False,
        "description": "Pure rule-based, no LLM calls"
    },
    
    "balanced": {
        "enabled": True,
        "method": "few-shot",
        "max_candidates_to_score": 3,
        "description": "Few-shot on top 3 candidates"
    },
    
    "accurate": {
        "enabled": True,
        "method": "cot",
        "max_candidates_to_score": 5,
        "description": "CoT reasoning on top 5 candidates"
    },
    
    "research": {
        "enabled": True,
        "method": "self-consistency",
        "max_candidates_to_score": 5,
        "self_consistency_samples": 5,
        "description": "Self-consistency with 5 samples"
    },

    "rl_optimized": {
        "enabled": True,
        "method": "cot",
        "max_candidates_to_score": 8,
        "use_rl_reranker": True,
        "rl_policy_path": "rl_policy.json",
        "rl_reward_weights": {
            "final_answer": 0.55,
            "reasoning_alignment": 0.30,
            "negation_consistency": 0.10,
            "calibration": 0.05
        },
        "description": "LLM scoring with RL reranking policy"
    }
}

# ==================== HELPER FUNCTIONS ====================

def get_scoring_config(profile: str = "balanced") -> dict:
    """Get configuration for a performance profile"""
    
    if profile not in PERFORMANCE_PROFILES:
        print(f"Warning: Unknown profile '{profile}', using 'balanced'")
        profile = "balanced"
    
    config = PERFORMANCE_PROFILES[profile].copy()
    
    # Add defaults from main config
    if "temperature" not in config:
        config["temperature"] = LLM_SCORING_CONFIG["temperature"]
    if "scoring_model" not in config:
        config["scoring_model"] = LLM_SCORING_CONFIG["scoring_model"]
    
    return config

def print_technique_info(method: str):
    """Print information about a prompting technique"""
    if method not in TECHNIQUE_INFO:
        print(f"Unknown method: {method}")
        return
    
    info = TECHNIQUE_INFO[method]
    
    print(f"\n{'='*60}")
    print(f"📚 {info['name']}")
    print('='*60)
    print(f"\n📝 Description: {info['description']}")
    print(f"\n✅ Pros:")
    for pro in info['pros']:
        print(f"   • {pro}")
    print(f"\n⚠️  Cons:")
    for con in info['cons']:
        print(f"   • {con}")
    print(f"\n🎯 Best for: {info['best_for']}")
    print(f"\n💡 Example:\n   {info['example']}")
    print('='*60)

def compare_methods():
    """Compare all prompting methods"""
    print("\n" + "="*80)
    print("🔬 PROMPTING TECHNIQUE COMPARISON")
    print("="*80)
    
    headers = ["Method", "Speed", "Accuracy", "Cost", "Best Use Case"]
    print(f"\n{headers[0]:<20} {headers[1]:<10} {headers[2]:<10} {headers[3]:<10} {headers[4]}")
    print("-" * 80)
    
    comparisons = [
        ("CoT", "Slow", "High", "High", "Complex cases"),
        ("ReAct", "Slow", "High", "High", "Verification needed"),
        ("Few-Shot", "Fast", "Medium", "Low", "Standard presentations"),
        ("Self-Consistency", "Very Slow", "Very High", "Very High", "High stakes"),
        ("Hybrid", "Medium", "High", "Medium", "Production use")
    ]
    
    for method, speed, accuracy, cost, use_case in comparisons:
        print(f"{method:<20} {speed:<10} {accuracy:<10} {cost:<10} {use_case}")
    
    print("\n" + "="*80)

# ==================== EXAMPLE USAGE ====================

def example_usage():
    """Show how to use different configurations"""
    
    print("\n" + "="*60)
    print("📖 EXAMPLE USAGE")
    print("="*60)
    
    print("\n1️⃣  Quick Setup (Balanced):")
    print("""
    from stage4_scoring_enhanced import EnhancedAttributeScorer
    
    scorer = EnhancedAttributeScorer(
        api_key=your_key,
        use_llm_scoring=True,
        llm_method="few-shot"  # Fast and effective
    )
    """)
    
    print("\n2️⃣  Maximum Accuracy:")
    print("""
    scorer = EnhancedAttributeScorer(
        api_key=your_key,
        use_llm_scoring=True,
        llm_method="cot"  # Deep reasoning
    )
    """)
    
    print("\n3️⃣  Using Configuration Profile:")
    print("""
    from config_llm_scoring import get_scoring_config
    
    config = get_scoring_config("accurate")
    
    scorer = EnhancedAttributeScorer(
        api_key=your_key,
        use_llm_scoring=config["enabled"],
        llm_method=config["method"]
    )
    """)
    
    print("\n4️⃣  Disable LLM (Pure Rules):")
    print("""
    scorer = EnhancedAttributeScorer(
        api_key=your_key,
        use_llm_scoring=False  # Fast, no API calls
    )
    """)

if __name__ == "__main__":
    print("="*80)
    print("🎯 LLM-BASED SCORING CONFIGURATION")
    print("="*80)
    
    # Show all techniques
    for method in TECHNIQUE_INFO.keys():
        print_technique_info(method)
    
    # Compare methods
    compare_methods()
    
    # Show usage
    example_usage()
    
    print("\n✅ Configuration loaded!")
    print("   Use get_scoring_config('profile_name') to get settings")