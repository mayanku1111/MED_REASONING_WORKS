
from pydantic import BaseModel, Field
from typing import List, Dict, Literal, Optional
from openrouter import OpenRouter
import json
import os
import re 
# ===== TRACE SCHEMAS =====

class EvidenceTrace(BaseModel):
    focus: Literal[
        "symptom", 
        "sign",          
        "lab",           
        "imaging",        
        "vital",         
        "modifier", 
        "radiation", 
        "negation", 
        "temporal", 
        "history"
    ]
    name: str
    attributes: Dict[str, str] = Field(default_factory=dict)
    certainty: float = Field(ge=0.0, le=1.0, default=0.8)
    source_sentence: Optional[str] = None

class HypothesisTrace(BaseModel):
    """Agent B: Diagnostic hypotheses"""
    hypothesis: str
    snomed_search_terms: List[str]  # What to search in SNOMED
    supporting_evidence: List[str]
    confidence: float = Field(ge=0.0, le=1.0)

class CounterfactualTrace(BaseModel):
    """Agent C: Contrastive reasoning"""
    hypothesis: str
    counterfactual_condition: str
    impact: Literal["supports", "contradicts", "neutral"]
    reasoning: str

# ===== AGENT PROMPTS =====

EXTRACTOR_SYSTEM = """You are a clinical fact extraction agent.

Extract ONLY factual clinical information that is EXPLICITLY stated in the text.
Do NOT infer, assume, or add any information not directly mentioned.
Do NOT diagnose.

For each finding EXPLICITLY mentioned, output:
- focus: symptom/sign/lab/imaging/vital/history/modifier/radiation/negation/temporal
- name: exact clinical term as stated
- attributes: {{duration, quality, location, severity, value}} - ONLY if mentioned
- certainty: 0.0-1.0 based on how explicitly it's stated
- source_sentence: the exact original sentence

Focus types:
- symptom: Patient-reported (e.g., "headache", "pain")
- sign: Objective findings (e.g., "weakness", "confusion")
- lab: Laboratory values (e.g., "CD4 count 45")
- imaging: Imaging findings (e.g., "ring-enhancing lesions")
- vital: Vital signs (e.g., "fever", "temperature")
- history: Past medical history (e.g., "untreated HIV")
- modifier: Qualifiers (e.g., "severe", "mild")
- temporal: Time information (e.g., "5 days", "acute")

CRITICAL: If a finding is NOT mentioned in the text, do NOT include it.

Output ONLY valid JSON in this exact format."""

EXTRACTOR_USER = """Extract clinical facts from this text. 
Extract ONLY what is EXPLICITLY stated. Do not add assumptions.

Clinical text:
{clinical_text}

Output JSON format:
{{
  "traces": [
    {{
      "focus": "symptom",
      "name": "headache",
      "attributes": {{"duration": "5 days"}},
      "certainty": 0.95,
      "source_sentence": "presents with headache, fever, and confusion for 5 days"
    }},
    {{
      "focus": "lab",
      "name": "CD4 count",
      "attributes": {{"value": "45 cells/µL"}},
      "certainty": 1.0,
      "source_sentence": "CD4 count is 45 cells/µL"
    }},
    {{
      "focus": "imaging",
      "name": "ring-enhancing lesions",
      "attributes": {{"location": "basal ganglia"}},
      "certainty": 1.0,
      "source_sentence": "MRI shows multiple ring-enhancing lesions in the basal ganglia"
    }}
  ]
}}

Remember: Extract ONLY what is explicitly mentioned. Do not infer symptoms."""

# ===== PLANNER AGENT =====

PLANNER_SYSTEM = """You are a differential diagnosis agent.

Given clinical evidence, generate plausible diagnostic hypotheses.

For each hypothesis:
- Provide the clinical diagnosis name
- Generate 2-4 SNOMED search terms (use clinical terminology)
- List supporting evidence from the case
- Assign confidence (0.0-1.0)

Generate 5-8 hypotheses ranked by clinical likelihood.

Output ONLY valid JSON."""

PLANNER_USER = """Generate differential diagnoses for this clinical presentation.

Clinical presentation:
{clinical_text}

Evidence found:
{evidence_traces}

Output JSON format:
{{
  "hypotheses": [
    {{
      "hypothesis": "Cerebral toxoplasmosis",
      "snomed_search_terms": ["toxoplasmosis", "cerebral toxoplasmosis", "toxoplasma encephalitis", "CNS toxoplasmosis"],
      "supporting_evidence": ["HIV", "CD4 count 45", "ring-enhancing lesions", "headache", "confusion"],
      "confidence": 0.90
    }},
    {{
      "hypothesis": "Primary CNS lymphoma",
      "snomed_search_terms": ["CNS lymphoma", "primary central nervous system lymphoma", "brain lymphoma"],
      "supporting_evidence": ["HIV", "low CD4", "brain lesions"],
      "confidence": 0.75
    }},
    {{
      "hypothesis": "Progressive multifocal leukoencephalopathy",
      "snomed_search_terms": ["PML", "progressive multifocal leukoencephalopathy", "JC virus"],
      "supporting_evidence": ["HIV", "immunosuppression", "neurologic symptoms"],
      "confidence": 0.60
    }}
  ]
}}

Generate 5-8 clinically relevant hypotheses based on the evidence."""

# ===== CONTRASTIVE AGENT =====

CONTRASTIVE_SYSTEM = """You are a counterfactual reasoning agent.

For each hypothesis, generate 2-4 counterfactual scenarios that would:
- Support the diagnosis (if findings were different)
- Contradict the diagnosis (if findings were present)
- Be neutral (alternative explanations)

Use clinical reasoning to assess impact.

Output ONLY valid JSON."""

CONTRASTIVE_USER = """Generate counterfactual reasoning for these hypotheses.

Hypotheses:
{hypotheses}

Clinical context:
{clinical_text}

For each hypothesis, generate 2-4 counterfactual scenarios that explore:
1. What findings would SUPPORT this diagnosis more strongly
2. What findings would CONTRADICT this diagnosis
3. What alternative explanations exist

Consider:
- Missing information that would be diagnostic
- Findings that rule in/out the condition
- Alternative diagnoses with similar presentations
- Treatment response predictions

Output JSON format:
{{
  "counterfactuals": [
    {{
      "hypothesis": "Cerebral toxoplasmosis",
      "counterfactual_condition": "If patient had been on prophylaxis with trimethoprim-sulfamethoxazole",
      "impact": "contradicts",
      "reasoning": "Prophylaxis significantly reduces toxoplasmosis risk in HIV patients with CD4 < 100"
    }},
    {{
      "hypothesis": "Cerebral toxoplasmosis",
      "counterfactual_condition": "If lesions showed periventricular location instead of basal ganglia",
      "impact": "supports",
      "reasoning": "Periventricular location is more typical of PML or CMV, less typical of toxoplasmosis which prefers basal ganglia"
    }},
    {{
      "hypothesis": "Primary CNS lymphoma",
      "counterfactual_condition": "If CSF showed elevated protein and positive EBV PCR",
      "impact": "supports",
      "reasoning": "EBV is strongly associated with CNS lymphoma in HIV patients"
    }},
    {{
      "hypothesis": "Primary CNS lymphoma",
      "counterfactual_condition": "If lesions showed rapid improvement with empiric toxoplasmosis treatment",
      "impact": "contradicts",
      "reasoning": "Lymphoma would not respond to anti-toxoplasma therapy; response suggests toxoplasmosis instead"
    }},
    {{
      "hypothesis": "Progressive multifocal leukoencephalopathy",
      "counterfactual_condition": "If MRI showed white matter lesions without mass effect or enhancement",
      "impact": "supports",
      "reasoning": "Non-enhancing white matter lesions are classic for PML, unlike the ring-enhancing lesions seen here"
    }},
    {{
      "hypothesis": "Tuberculous meningitis",
      "counterfactual_condition": "If patient had recent PPD positive test and basal meningeal enhancement on MRI",
      "impact": "supports",
      "reasoning": "TB meningitis can present with ring-enhancing lesions (tuberculomas) in immunocompromised patients"
    }}
  ]
}}

Generate 2-4 clinically relevant counterfactuals for EACH hypothesis provided."""



# ===== MULTI-AGENT TRACE GENERATOR =====

class MultiAgentTraceGenerator:
    """Stage 1: Orchestrates all three agents"""
    
    def __init__(self, api_key: str, model: str = "google/gemini-2.5-flash-lite"):
        self.api_key = api_key
        self.model = model
    
    def _call_llm(self, system: str, user: str, effort: str = "medium") -> dict:
        """Helper to call OpenRouter with JSON output"""
        try:
            with OpenRouter(api_key=self.api_key) as client:
                response = client.chat.send(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user}
                    ],
                    # REMOVED: response_format (not supported by free models)
                    temperature=0.3  # Slightly higher for better extraction
                )
                
                content = response.choices[0].message.content
                
                print(f"\n🤖 LLM Raw Response:")
                print(f"   {content[:300]}...")
                
                if not content or not content.strip():
                    print(f"⚠️  Empty response from LLM")
                    return {"traces": [], "hypotheses": [], "counterfactuals": []}
                
                # Try to extract JSON from markdown code blocks
                json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
                if json_match:
                    content = json_match.group(1)
                
                # Try to parse JSON
                try:
                    parsed = json.loads(content)
                    print(f"   ✓ Parsed JSON successfully")
                    return parsed
                except json.JSONDecodeError as e:
                    print(f"⚠️  Invalid JSON from LLM:")
                    print(f"   Error: {e}")
                    
                    # Try to find ANY JSON object in the response
                    json_obj_match = re.search(r'\{[^{}]*"traces"[^{}]*\[.*?\][^{}]*\}', content, re.DOTALL)
                    if json_obj_match:
                        try:
                            return json.loads(json_obj_match.group(0))
                        except:
                            pass
                    
                    return {"traces": [], "hypotheses": [], "counterfactuals": []}
                    
        except Exception as e:
            print(f"❌ LLM API Error: {e}")
            import traceback
            traceback.print_exc()
            return {"traces": [], "hypotheses": [], "counterfactuals": []}
    
    def extract_evidence(self, clinical_text: str, preprocessed_sentences: list) -> List[EvidenceTrace]:
        """Extract evidence traces from clinical text"""
        
        prompt_text = " ".join([s.text for s in preprocessed_sentences]) if preprocessed_sentences else clinical_text
        
        result = self._call_llm(
            EXTRACTOR_SYSTEM,
            EXTRACTOR_USER.format(clinical_text=prompt_text),
        )
        
        # DEBUG: Print what we got back
        print(f"\n🔍 DEBUG - LLM Response:")
        print(f"   Raw result: {result}")
        print(f"   Number of traces: {len(result.get('traces', []))}")
        
        # Handle error response
        if "error" in result:
            print(f"⚠️  Extraction error: {result['error']}")
            return []
        
        traces = []
        for idx, trace_data in enumerate(result.get("traces", [])):
            try:
                trace = EvidenceTrace(**trace_data)
                traces.append(trace)
                print(f"   ✓ Trace {idx+1}: {trace.name} ({trace.focus})")
            except Exception as e:
                print(f"   ❌ Failed to parse trace {idx+1}: {e}")
                print(f"      Data: {trace_data}")
                continue
        
        if not traces:
            print(f"⚠️  WARNING: No valid traces extracted from:")
            print(f"   {prompt_text[:200]}...")
        
        return traces
    
    def generate_hypotheses(self, clinical_text: str, evidence_traces: List[EvidenceTrace]) -> List[HypothesisTrace]:
        """Agent B: Hypothesis generation"""
        print("💡 Agent B: Generating hypotheses...")
        
        evidence_summary = "\n".join([
            f"- {t.name} (certainty: {t.certainty}, focus: {t.focus})"
            for t in evidence_traces
        ])
        
        result = self._call_llm(
            PLANNER_SYSTEM,
            PLANNER_USER.format(
                clinical_text=clinical_text,
                evidence_traces=evidence_summary
            ),
            effort="medium"
        )
        
        hypotheses = [HypothesisTrace(**h) for h in result.get("hypotheses", [])]
        print(f"   ✓ Generated {len(hypotheses)} hypotheses")
        return hypotheses
    
    def generate_counterfactuals(
        self, 
        hypotheses: List[HypothesisTrace],
        clinical_text: str  # ADD THIS PARAMETER
    ) -> List[CounterfactualTrace]:
        """Agent C: Generate counterfactual reasoning"""
        if not hypotheses:
            return []
        
        # Format hypotheses for prompt
        hyp_summary = "\n".join([
            f"- {h.hypothesis} (confidence: {h.confidence})"
            for h in hypotheses
        ])
        
        result = self._call_llm(
            CONTRASTIVE_SYSTEM,
            CONTRASTIVE_USER.format(
                hypotheses=hyp_summary,
                clinical_text=clinical_text  # ADD THIS
            ),
        )
        
        counterfactuals = []
        for cf_data in result.get("counterfactuals", []):
            try:
                cf = CounterfactualTrace(**cf_data)
                counterfactuals.append(cf)
            except Exception as e:
                print(f"⚠️  Could not parse counterfactual: {e}")
                continue
        
        return counterfactuals
    
    def generate_all_traces(
        self, 
        clinical_text: str, 
        preprocessed_sentences: List
    ) -> Dict[str, List]:
        """Run all three agents"""
        
        print("🔍 Agent A: Extracting evidence traces...")
        evidence = self.extract_evidence(clinical_text, preprocessed_sentences)
        print(f"   ✓ Extracted {len(evidence)} evidence traces")
        
        print("💡 Agent B: Generating hypotheses...")
        hypotheses = self.generate_hypotheses(clinical_text, evidence)
        print(f"   ✓ Generated {len(hypotheses)} hypotheses")
        
        print("🔀 Agent C: Generating counterfactuals...")
        counterfactuals = self.generate_counterfactuals(
            hypotheses, 
            clinical_text  # ADD THIS ARGUMENT
        )
        print(f"   ✓ Generated {len(counterfactuals)} counterfactuals")
        
        return {
            "evidence": evidence,
            "hypotheses": hypotheses,
            "counterfactuals": counterfactuals
        }

def test_trace_generation():
    """Test trace generation"""
    from stage0_preprocessing import ClinicalPreprocessor
    
    preprocessor = ClinicalPreprocessor()
    generator = MultiAgentTraceGenerator(api_key=os.getenv("OPENROUTER_API_KEY"))
    
    test_text = """
    55-year-old male with tight central chest pain for 2 days.
    Pain worsens with exertion, especially climbing stairs.
    Radiates to left arm and jaw.
    Associated with sweating and shortness of breath.
    No fever. No cough.
    History of hypertension and smoking.
    """
    
    preprocessed = preprocessor.preprocess(test_text)
    traces = generator.generate_all_traces(test_text, preprocessed)
    
    print("\n=== EVIDENCE TRACES ===")
    for t in traces["evidence"]:
        print(f"- {t.name} [{t.focus}] (certainty: {t.certainty})")
        print(f"  Attributes: {t.attributes}")
    
    print("\n=== HYPOTHESES ===")
    for h in traces["hypotheses"]:
        print(f"- {h.hypothesis} (confidence: {h.confidence})")
        print(f"  Search terms: {h.snomed_search_terms}")
    
    print("\n=== COUNTERFACTUALS ===")
    for c in traces["counterfactuals"]:
        print(f"- {c.hypothesis}: {c.counterfactual_condition}")
        print(f"  Impact: {c.impact}")

if __name__ == "__main__":
    test_trace_generation()
