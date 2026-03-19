"""
Main Pipeline with Enhanced LLM-Based Scoring
"""

from stage0_preprocessing import ClinicalPreprocessor
from stage1_traces import MultiAgentTraceGenerator
from stage2_normalization import TraceNormalizer
from stage3_retrieval import SNOMEDRetriever
from stage4_scoring import EnhancedAttributeScorer  
from stage6_adjudication import LLMAdjudicator
from stage8_generation import GroundedAnswerGenerator
from config_llm_scoring import get_scoring_config  
from typing import Dict
import os
import time

class EnhancedTraceToSNOMEDPipeline:
    """Pipeline with LLM-based intelligent scoring"""
    
    def __init__(self, 
                 openrouter_api_key: str,
                 snomed_base_url: str = "http://localhost:8080",
                 model: str = "google/gemini-2.5-flash-lite",
                 use_normalization: bool = True,
                 scoring_profile: str = "balanced"):  # NEW: fast, balanced, accurate, research
        
        # Initialize all stages
        self.preprocessor = ClinicalPreprocessor()
        self.trace_generator = MultiAgentTraceGenerator(openrouter_api_key, model)
        self.normalizer = TraceNormalizer() if use_normalization else None
        self.retriever = SNOMEDRetriever(
            base_url=snomed_base_url,
            api_key=openrouter_api_key, 
            model=model
        )
        
        # === ENHANCED SCORER ===
        scoring_config = get_scoring_config(scoring_profile)
        self.scorer = EnhancedAttributeScorer(
            api_key=openrouter_api_key,
            model=scoring_config.get("scoring_model", model),
            use_llm_scoring=scoring_config.get("enabled", False),
            llm_method=scoring_config.get("method", "cot"),
            use_rl_reranker=scoring_config.get("use_rl_reranker", False),
            rl_policy_path=scoring_config.get("rl_policy_path"),
            rl_reward_weights=scoring_config.get("rl_reward_weights"),
        )
        
        self.adjudicator = LLMAdjudicator(openrouter_api_key, model)
        self.answer_generator = GroundedAnswerGenerator(openrouter_api_key, model)
        
        self.use_normalization = use_normalization
        self.scoring_profile = scoring_profile
        
        print(f"🎯 Scoring Profile: {scoring_profile.upper()}")
        if scoring_config.get("enabled"):
            print(f"   Method: {scoring_config.get('method', 'unknown').upper()}")
            print(f"   Max candidates: {scoring_config.get('max_candidates_to_score', 5)}")
        else:
            print(f"   Method: RULE-BASED ONLY (no LLM)")
    
    def run(self, clinical_query: str) -> Dict:
        """Execute complete pipeline"""
        
        print("\n" + "="*60)
        print("🏥 ENHANCED TRACE-TO-SNOMED PIPELINE")
        print("="*60)
        print(f"\nInput Query:\n{clinical_query}\n")
        print("="*60)
        
        results = {
            "input": clinical_query,
            "stages": {},
            "config": {
                "scoring_profile": self.scoring_profile,
                "normalization": self.use_normalization
            }
        }
        
        start_time = time.time()
        
        # STAGE 0: Preprocessing
        print("\n📋 STAGE 0: PREPROCESSING")
        print("-" * 60)
        preprocessed = self.preprocessor.preprocess(clinical_query)
        results["stages"]["stage0_preprocessing"] = {
            "sentences": [
                {
                    "text": s.text,
                    "type": s.sentence_type,
                    "is_negated": s.is_negated,
                    "temporal_info": s.temporal_info
                }
                for s in preprocessed
            ]
        }
        print(f"✓ Processed {len(preprocessed)} sentences")
        
        # STAGE 1: Multi-Agent Trace Generation
        print("\n🤖 STAGE 1: MULTI-AGENT TRACE GENERATION")
        print("-" * 60)
        traces = self.trace_generator.generate_all_traces(clinical_query, preprocessed)
        results["stages"]["stage1_traces"] = {
            "evidence": [t.dict() for t in traces["evidence"]],
            "hypotheses": [h.dict() for h in traces["hypotheses"]],
            "counterfactuals": [c.dict() for c in traces["counterfactuals"]]
        }
        print(f"✓ Generated {len(traces['evidence'])} evidence traces")
        print(f"✓ Generated {len(traces['hypotheses'])} hypotheses")
        print(f"✓ Generated {len(traces['counterfactuals'])} counterfactuals")
        
        # STAGE 2: Trace Normalization (optional)
        if self.use_normalization:
            print("\n🔄 STAGE 2: TRACE NORMALIZATION")
            print("-" * 60)
            canonical_traces = self.normalizer.normalize_all_traces(traces)
            results["stages"]["stage2_normalization"] = {
                "evidence": [
                    {
                        "trace_id": t.trace_id,
                        "name": t.name,
                        "focus": t.focus,
                        "attributes": t.attributes,
                        "certainty": t.certainty,
                        "is_negated": t.is_negated
                    }
                    for t in canonical_traces["evidence"]
                ],
                # ADD THESE COUNTS:
                "original_evidence_count": len(traces["evidence"]),
                "original_hypothesis_count": len(traces["hypotheses"]),
                "total_canonical_traces": sum(len(v) for v in canonical_traces.values()),
                "sample_normalized": [
                    {
                        "trace_id": t.trace_id,
                        "name": t.name,
                        "trace_type": t.trace_type,
                        "focus": t.focus
                    }
                    for t in canonical_traces["evidence"][:5]
                ]
            }
            traces_for_retrieval = traces
        else:
            print("\n⏭️  STAGE 2: NORMALIZATION SKIPPED")
            results["stages"]["stage2_normalization"] = {"skipped": True}
            traces_for_retrieval = traces
        
        # STAGE 3: SNOMED Retrieval
        print("\n🔎 STAGE 3: SNOMED CT RETRIEVAL")
        print("-" * 60)
        snomed_candidates = self.retriever.retrieve_for_traces(
            traces_for_retrieval["evidence"], 
            traces_for_retrieval["hypotheses"]
        )
        
        # Format for results output with by_trace structure
        retrieval_results = {
            "total_candidates": sum(len(cands) for cands in snomed_candidates.values()),
            "by_trace": {}  # NEW: Add this structure for Streamlit
        }
        
        # Organize by trace for better display
        for trace_key, candidates in snomed_candidates.items():
            retrieval_results["by_trace"][trace_key] = [
                {
                    "id": c.concept_id,
                    "name": c.fsn,
                    "strategy": c.retrieval_strategy
                }
                for c in candidates[:10]  # Limit to top 10 per trace for output
            ]
        
        results["stages"]["stage3_retrieval"] = retrieval_results
        print(f"✓ Retrieved {retrieval_results['total_candidates']} total SNOMED candidates")
        
        # STAGE 4-5: ENHANCED SCORING (with LLM!)
        print("\n📊 STAGE 4-5: ENHANCED SCORING & AGGREGATION")
        print("-" * 60)
        scored_concepts = self.scorer.aggregate_scores(
            snomed_candidates,
            traces_for_retrieval["evidence"],
            traces_for_retrieval["hypotheses"],
            traces_for_retrieval["counterfactuals"]
        )
        
        valid_concepts = [c for c in scored_concepts if c.total_score > -float('inf')]
        results["stages"]["stage4_scoring"] = {
            "scored_concepts": [
                {
                    "concept_id": c.concept_id,
                    "fsn": c.fsn,
                    "total_score": c.total_score,
                    "supporting_traces": c.supporting_traces,
                    "contradicting_traces": c.contradicting_traces,
                    "score_breakdown": c.score_breakdown,
                    "llm_reasoning": c.llm_reasoning  # NEW
                }
                for c in valid_concepts[:10]
            ]
        }
        print(f"✓ Scored {len(valid_concepts)} valid concepts")
        
        if not valid_concepts:
            print("⚠ No valid concepts found!")
            return results
        
        # Show top 3 with LLM reasoning
        print("\nTop 3 Concepts:")
        for i, c in enumerate(valid_concepts[:3], 1):
            print(f"  {i}. {c.fsn}")
            print(f"     Score: {c.total_score:.2f}")
            if c.llm_reasoning:
                print(f"     🧠 LLM: {c.llm_reasoning[:80]}...")
        
        # STAGE 6: Adjudication (if needed)
        adjudication_result = None
        final_concept = valid_concepts[0]
        
        if self.adjudicator.needs_adjudication(valid_concepts):
            print("\n⚖️  STAGE 6: LLM ADJUDICATION")
            print("-" * 60)
            adjudication_result = self.adjudicator.adjudicate(
                valid_concepts[0],
                valid_concepts[1],
                traces_for_retrieval["evidence"],
                n_samples=5
            )
            final_concept = adjudication_result["final_concept"]
            results["stages"]["stage6_adjudication"] = {
                "was_needed": True,
                "winner": adjudication_result["winner"],
                "votes": dict(adjudication_result["votes"]),
                "vote_ratio": adjudication_result["vote_ratio"]
            }
        else:
            print("\n✓ STAGE 6: ADJUDICATION SKIPPED")
            results["stages"]["stage6_adjudication"] = {"was_needed": False}
        
        # STAGE 8: Grounded Answer Generation
        print("\n📝 STAGE 8: GROUNDED ANSWER GENERATION")
        print("-" * 60)
        final_answer = self.answer_generator.generate_answer(
            final_concept,
            traces_for_retrieval["evidence"],
            adjudication_result
        )
        
        results["stages"]["stage8_answer"] = final_answer
        
        # Final summary
        elapsed = time.time() - start_time
        results["metadata"] = {
            "elapsed_time_seconds": elapsed,
            "total_stages": 8,
            "scoring_profile": self.scoring_profile
        }
        
        print("\n" + "="*60)
        print("✅ PIPELINE COMPLETE")
        print("="*60)
        print(f"\n🎯 FINAL SNOMED CONCEPT: {final_answer['snomed_concept']['name']}")
        print(f"   ID: {final_answer['snomed_concept']['id']}")
        print(f"   Score: {final_answer['snomed_concept']['score']:.2f}")
        print(f"\n💡 EXPLANATION:\n   {final_answer['explanation']}")
        print(f"\n⏱️  Completed in {elapsed:.1f}s")
        print("="*60)
        
        return results

def main():
    """Test with different scoring profiles"""
    
    test_query = """
    62-year-old woman with sudden right-sided weakness and slurred speech 
    started 3 hours ago. Face drooping on right. Cannot lift right arm.
    History of atrial fibrillation. Stopped warfarin 2 weeks ago.
    No headache. No seizures. BP 168/95, HR irregular at 110.
    """
    
    # Test different profiles
    profiles = ["fast", "balanced", "accurate"]
    
    for profile in profiles:
        print("\n\n" + "="*80)
        print(f"🧪 TESTING WITH PROFILE: {profile.upper()}")
        print("="*80)
        
        pipeline = EnhancedTraceToSNOMEDPipeline(
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
            scoring_profile=profile
        )
        
        results = pipeline.run(test_query)
        
        # Save results
        import json
        with open(f"results_{profile}.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\n📄 Results saved to results_{profile}.json")
        
        # Wait between profiles
        if profile != profiles[-1]:
            print("\n⏸️  Waiting 10 seconds before next profile...")
            import time
            time.sleep(10)

if __name__ == "__main__":
    main()