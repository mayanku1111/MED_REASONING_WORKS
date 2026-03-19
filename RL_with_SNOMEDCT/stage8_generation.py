
from openrouter import OpenRouter
from typing import List, Dict
import json

GROUNDING_SYSTEM = """You are a medical explanation assistant.

Generate a patient-friendly clinical explanation based ONLY on:
- Verified SNOMED concept(s)
- Supporting clinical evidence
- Explicit reasoning traces

CRITICAL RULES:
- Do NOT add information not present in the provided context
- Do NOT hallucinate symptoms or findings
- Do NOT diagnose - explain what the verified concept means
- Use clear, accessible language
- Include appropriate caveats about seeking professional care

Output ONLY valid JSON."""

GROUNDING_USER = """Generate a clinical explanation.

VERIFIED SNOMED CONCEPT:
ID: {concept_id}
Name: {concept_name}

SUPPORTING EVIDENCE:
{evidence}

REASONING PROVENANCE:
{provenance}

Output JSON format:
{{
  "explanation": "2-3 sentence clinical explanation",
  "key_findings": ["finding 1", "finding 2", ...],
  "clinical_significance": "brief statement",
  "recommendation": "appropriate next steps",
  "citations": ["trace1", "trace2", ...]
}}"""

class GroundedAnswerGenerator:
    """Stage 8: Generate verifiable, grounded answers"""
    
    def __init__(self, api_key: str, model: str = "google/gemini-2.5-flash-lite"):
        self.api_key = api_key
        self.model = model
    
    def format_evidence(self, evidence_traces) -> str:
        """Format evidence for prompt"""
        lines = []
        for t in evidence_traces:
            line = f"- {t.name} [{t.focus}]"
            if t.attributes:
                attrs = ", ".join([f"{k}:{v}" for k, v in t.attributes.items() if v])
                line += f" ({attrs})"
            line += f" [certainty: {t.certainty}]"
            lines.append(line)
        return "\n".join(lines)
    
    def format_provenance(self, concept_score) -> str:
        """Format scoring provenance"""
        lines = [f"Total Score: {concept_score.total_score:.2f}"]
        
        if concept_score.supporting_traces:
            lines.append(f"Supporting: {', '.join(concept_score.supporting_traces)}")
        
        if concept_score.contradicting_traces:
            lines.append(f"Contradicting: {', '.join(concept_score.contradicting_traces)}")
        
        if concept_score.score_breakdown:
            lines.append("Score breakdown:")
            for key, score in concept_score.score_breakdown.items():
                lines.append(f"  - {key}: {score:.2f}")
        
        return "\n".join(lines)
    
    def generate_answer(self, 
                       final_concept,
                       evidence_traces,
                       adjudication_result: Dict = None) -> Dict:
        """Generate grounded explanation"""
        
        print("📝 Stage 8: Generating grounded answer...")
        
        evidence_text = self.format_evidence(evidence_traces)
        provenance_text = self.format_provenance(final_concept)
        
        # Add adjudication info if present
        if adjudication_result:
            provenance_text += f"\n\nAdjudication: Winner by {adjudication_result['vote_ratio']:.1%} consensus"
        
        with OpenRouter(api_key=self.api_key) as client:
            response = client.chat.send(
                model=self.model,
                messages=[
                    {"role": "system", "content": GROUNDING_SYSTEM},
                    {"role": "user", "content": GROUNDING_USER.format(
                        concept_id=final_concept.concept_id,
                        concept_name=final_concept.fsn,
                        evidence=evidence_text,
                        provenance=provenance_text
                    )}
                ],
                response_format={"type": "json_object"},
                reasoning={"enabled": True, "effort": "medium"},
                temperature=0.3
            )
            
            result = json.loads(response.choices[0].message.content)
        
        print("   ✓ Generated grounded answer")
        
        # Package complete response
        return {
            "snomed_concept": {
                "id": final_concept.concept_id,
                "name": final_concept.fsn,
                "score": final_concept.total_score
            },
            "explanation": result.get("explanation", ""),
            "key_findings": result.get("key_findings", []),
            "clinical_significance": result.get("clinical_significance", ""),
            "recommendation": result.get("recommendation", ""),
            "provenance": {
                "supporting_traces": final_concept.supporting_traces,
                "contradicting_traces": final_concept.contradicting_traces,
                "score_breakdown": final_concept.score_breakdown,
                "adjudication": adjudication_result if adjudication_result else None
            }
        }

def test_generation():
    """Test answer generation"""
    from stage4_scoring import ConceptScore
    from stage1_traces import EvidenceTrace
    import os
    
    generator = GroundedAnswerGenerator(api_key=os.getenv("OPENROUTER_API_KEY"))
    
    concept = ConceptScore(
        concept_id="194828000",
        fsn="Angina pectoris (disorder)",
        total_score=5.2,
        supporting_traces=["chest pain", "exertional", "arm radiation"],
        score_breakdown={
            "evidence:chest pain": 1.8,
            "evidence:exertional": 1.5,
            "hypothesis:angina": 2.0
        }
    )
    
    evidence = [
        EvidenceTrace(
            focus="symptom",
            name="chest pain",
            attributes={"quality": "tight", "location": "central"},
            certainty=0.9
        ),
        EvidenceTrace(
            focus="modifier",
            name="exertional",
            attributes={"trigger": "stairs"},
            certainty=0.8
        )
    ]
    
    result = generator.generate_answer(concept, evidence)
    
    print("\n=== GENERATED ANSWER ===")
    print(f"Concept: {result['snomed_concept']['name']}")
    print(f"\nExplanation: {result['explanation']}")
    print(f"\nKey Findings: {result['key_findings']}")
    print(f"\nRecommendation: {result['recommendation']}")

if __name__ == "__main__":
    test_generation()