import re
from openrouter import OpenRouter
from typing import List, Dict
import json
from collections import Counter

ADJUDICATION_SYSTEM = """You are a medical reasoning expert adjudicating between two SNOMED concepts.

Given clinical evidence, determine which concept better fits.

CRITICAL RULES:
- Consider ALL evidence carefully
- Explain your reasoning step-by-step
- Choose based on clinical fit, not concept name familiarity
- Output ONLY valid JSON - no markdown, no code blocks, no extra text

Start your response with { and end with }"""

ADJUDICATION_USER = """Choose the better matching SNOMED concept.

CLINICAL EVIDENCE:
{evidence}

CONCEPT A:
ID: {concept_a_id}
Name: {concept_a_name}
Supporting traces: {concept_a_support}

CONCEPT B:
ID: {concept_b_id}
Name: {concept_b_name}
Supporting traces: {concept_b_support}

Think step-by-step:
1. Which concept explains MORE of the evidence?
2. Are there any contradictions?
3. Which is more clinically appropriate?

Output ONLY this JSON (no markdown):
{{
  "choice": "A" or "B",
  "reasoning": "step-by-step explanation in 2-3 sentences",
  "confidence": 0.0-1.0,
  "key_deciding_factor": "the main reason for this choice"
}}"""

class LLMAdjudicator:
    """Stage 6: Resolve ambiguous cases using LLM self-consistency"""
    
    def __init__(self, api_key: str, model: str = "google/gemini-2.5-flash-lite"):
        self.api_key = api_key
        self.model = model
    
    def needs_adjudication(self, top_concepts, threshold: float = 0.3) -> bool:
        """Check if top concepts are too close"""
        if len(top_concepts) < 2:
            return False
        
        score_diff = abs(top_concepts[0].total_score - top_concepts[1].total_score)
        return score_diff < threshold
    
    def _extract_json_from_response(self, content: str) -> dict:
        """Extract JSON from LLM response, handling markdown code blocks"""
        content = content.strip()
        
        # Try to extract from markdown code blocks
        if "```json" in content:
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', content, re.DOTALL)
            if json_match:
                content = json_match.group(1)
        elif "```" in content:
            json_match = re.search(r'```\s*(\{.*?\})\s*```', content, re.DOTALL)
            if json_match:
                content = json_match.group(1)
        
        # Remove any remaining markdown or extra text
        # Find the first { and last }
        start_idx = content.find('{')
        end_idx = content.rfind('}')
        
        if start_idx != -1 and end_idx != -1:
            content = content[start_idx:end_idx+1]
        
        return json.loads(content)
    
    def adjudicate(self, 
                   concept_a, 
                   concept_b, 
                   evidence_traces,
                   n_samples: int = 5) -> Dict:
        """Run self-consistency voting"""
        
        print(f"⚖️  Stage 6: Adjudicating between close concepts...")
        print(f"   Concept A: {concept_a.fsn} (score: {concept_a.total_score:.2f})")
        print(f"   Concept B: {concept_b.fsn} (score: {concept_b.total_score:.2f})")
        
        # Format evidence
        evidence_text = "\n".join([
            f"- {t.name} [{t.focus}] (certainty: {t.certainty})"
            + (f" | Attributes: {t.attributes}" if t.attributes else "")
            for t in evidence_traces
        ])
        
        votes = []
        reasonings = []
        
        # Use direct HTTP requests for better control
        import requests
        
        for i in range(n_samples):
            print(f"   Vote {i+1}/{n_samples}...", end=" ")
            
            try:
                response = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "http://localhost:8501",
                    },
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": ADJUDICATION_SYSTEM},
                            {"role": "user", "content": ADJUDICATION_USER.format(
                                evidence=evidence_text,
                                concept_a_id=concept_a.concept_id,
                                concept_a_name=concept_a.fsn,
                                concept_a_support=", ".join(concept_a.supporting_traces[:5]),  # Limit length
                                concept_b_id=concept_b.concept_id,
                                concept_b_name=concept_b.fsn,
                                concept_b_support=", ".join(concept_b.supporting_traces[:5])
                            )}
                        ],
                        "temperature": 0.7,  # Add variance for self-consistency
                        "max_tokens": 1000
                    },
                    timeout=30
                )
                
                response.raise_for_status()
                result_json = response.json()
                
                # Extract content
                content = result_json['choices'][0]['message']['content']
                
                # Debug output for first vote
                if i == 0:
                    print(f"\n   🔍 Sample response: {content[:200]}...")
                
                # Parse JSON
                result = self._extract_json_from_response(content)
                
                vote = result.get("choice", "A").upper()
                reasoning = result.get("reasoning", "No reasoning provided")
                
                # Validate vote
                if vote not in ["A", "B"]:
                    print(f"⚠️ Invalid vote '{vote}', defaulting to A")
                    vote = "A"
                
                votes.append(vote)
                reasonings.append(reasoning)
                print(f"✓ {vote}")
                
            except json.JSONDecodeError as e:
                print(f"✗ JSON Error: {e}")
                if 'content' in locals():
                    print(f"   Raw: {content[:200]}")
                votes.append("A")  # Default fallback
                reasonings.append("Error parsing JSON response")
                
            except requests.exceptions.RequestException as e:
                print(f"✗ API Error: {e}")
                votes.append("A")
                reasonings.append("API request failed")
                
            except Exception as e:
                print(f"✗ Unexpected error: {e}")
                votes.append("A")
                reasonings.append(f"Error: {str(e)}")
        
        # Count votes
        vote_counts = Counter(votes)
        winner = vote_counts.most_common(1)[0][0]
        vote_ratio = vote_counts[winner] / len(votes)
        
        print(f"\n   🏆 Winner: Concept {winner}")
        print(f"      Votes: {vote_counts['A']} for A, {vote_counts['B']} for B")
        print(f"      Confidence: {vote_ratio:.1%}")
        
        final_concept = concept_a if winner == "A" else concept_b
        print(f"      Selected: {final_concept.fsn}")
        
        return {
            "winner": winner,
            "votes": vote_counts,
            "vote_ratio": vote_ratio,
            "reasonings": reasonings,
            "final_concept": final_concept
        }

def test_adjudication():
    """Test adjudication"""
    from stage4_scoring import ConceptScore
    from stage1_traces import EvidenceTrace
    import os
    
    adjudicator = LLMAdjudicator(api_key=os.getenv("OPENROUTER_API_KEY"))
    
    # Mock close concepts
    concept_a = ConceptScore(
        concept_id="26929004",
        fsn="Alzheimer's disease (disorder)",
        total_score=13.5,
        supporting_traces=["memory loss", "confusion", "cognitive decline"]
    )
    
    concept_b = ConceptScore(
        concept_id="429998004",
        fsn="Vascular dementia (disorder)",
        total_score=13.5,
        supporting_traces=["memory loss", "stroke history", "stepwise decline"]
    )
    
    evidence = [
        EvidenceTrace(
            focus="symptom",
            name="memory loss",
            attributes={"onset": "gradual", "duration": "2 years"},
            certainty=0.95
        ),
        EvidenceTrace(
            focus="sign",
            name="confusion",
            attributes={"severity": "moderate"},
            certainty=0.9
        ),
        EvidenceTrace(
            focus="history",
            name="hypertension",
            attributes={},
            certainty=0.8
        )
    ]
    
    print("\n" + "="*70)
    print("TEST: Adjudication Between Close Concepts")
    print("="*70)
    
    result = adjudicator.adjudicate(concept_a, concept_b, evidence, n_samples=3)
    
    print(f"\n{'='*70}")
    print(f"Final Decision: {result['final_concept'].fsn}")
    print(f"Vote Confidence: {result['vote_ratio']:.1%}")
    print(f"\nSample Reasoning:")
    print(f"  {result['reasonings'][0]}")
    print("="*70)

if __name__ == "__main__":
    test_adjudication()