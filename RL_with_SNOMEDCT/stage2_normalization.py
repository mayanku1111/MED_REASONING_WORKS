
from typing import List, Dict, Set
from dataclasses import dataclass, field
import hashlib

@dataclass
class CanonicalTrace:
    """Normalized, canonical trace format"""
    trace_id: str  # Unique identifier
    trace_type: str  # evidence, hypothesis, counterfactual
    focus: str  # symptom, modifier, negation, etc.
    name: str  # Normalized name
    attributes: Dict[str, str] = field(default_factory=dict)
    certainty: float = 1.0
    source_traces: List[str] = field(default_factory=list)  # Provenance
    is_negated: bool = False
    
class TraceNormalizer:
    """Stage 2: Normalizes all traces into canonical format"""
    
    def __init__(self):
        self.seen_traces: Set[str] = set()
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text: lowercase, strip, remove extra spaces"""
        return " ".join(text.lower().strip().split())
    
    def _generate_trace_id(self, trace_type: str, name: str, focus: str) -> str:
        """Generate unique trace ID"""
        # Hash based on type + normalized name + focus
        content = f"{trace_type}:{self._normalize_text(name)}:{focus}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def _merge_similar_traces(self, traces: List[CanonicalTrace]) -> List[CanonicalTrace]:
        """Merge duplicate or very similar traces"""
        merged = {}
        
        for trace in traces:
            key = f"{trace.focus}:{trace.name}"
            
            if key in merged:
                # Merge: take max certainty, combine attributes
                existing = merged[key]
                existing.certainty = max(existing.certainty, trace.certainty)
                existing.attributes.update(trace.attributes)
                existing.source_traces.extend(trace.source_traces)
            else:
                merged[key] = trace
        
        return list(merged.values())
    
    def normalize_evidence_traces(self, evidence_traces) -> List[CanonicalTrace]:
        """Convert evidence traces to canonical format"""
        canonical_traces = []
        
        for trace in evidence_traces:
            # Normalize name
            normalized_name = self._normalize_text(trace.name)
            
            # Normalize attributes
            normalized_attrs = {
                self._normalize_text(k): self._normalize_text(str(v))
                for k, v in trace.attributes.items()
                if v  # Skip empty values
            }
            
            # Create canonical trace
            canonical = CanonicalTrace(
                trace_id=self._generate_trace_id("evidence", normalized_name, trace.focus),
                trace_type="evidence",
                focus=trace.focus,
                name=normalized_name,
                attributes=normalized_attrs,
                certainty=trace.certainty,
                source_traces=[trace.source_sentence] if trace.source_sentence else [],
                is_negated=(trace.focus == "negation")
            )
            
            canonical_traces.append(canonical)
        
        return canonical_traces
    
    def normalize_hypothesis_traces(self, hypothesis_traces) -> List[CanonicalTrace]:
        """Convert hypothesis traces to canonical format"""
        canonical_traces = []
        
        for hyp in hypothesis_traces:
            normalized_name = self._normalize_text(hyp.hypothesis)
            
            # Create canonical hypothesis trace
            canonical = CanonicalTrace(
                trace_id=self._generate_trace_id("hypothesis", normalized_name, "diagnosis"),
                trace_type="hypothesis",
                focus="diagnosis",
                name=normalized_name,
                attributes={
                    "snomed_search_terms": ",".join(hyp.snomed_search_terms),
                    "supporting_evidence": ",".join(hyp.supporting_evidence)
                },
                certainty=hyp.confidence,
                source_traces=["hypothesis_planner"]
            )
            
            canonical_traces.append(canonical)
        
        return canonical_traces
    
    def normalize_counterfactual_traces(self, counterfactual_traces) -> List[CanonicalTrace]:
        """Convert counterfactual traces to canonical format"""
        canonical_traces = []
        
        for cf in counterfactual_traces:
            normalized_name = self._normalize_text(cf.hypothesis)
            normalized_condition = self._normalize_text(cf.counterfactual_condition)
            
            # Create canonical counterfactual trace
            canonical = CanonicalTrace(
                trace_id=self._generate_trace_id("counterfactual", normalized_name, "constraint"),
                trace_type="counterfactual",
                focus="constraint",
                name=normalized_name,
                attributes={
                    "condition": normalized_condition,
                    "impact": cf.impact,
                    "reasoning": self._normalize_text(cf.reasoning)
                },
                certainty=1.0,  # Counterfactuals are logical constraints
                source_traces=["counterfactual_generator"]
            )
            
            canonical_traces.append(canonical)
        
        return canonical_traces
    
    def validate_traces(self, traces: List[CanonicalTrace]) -> List[CanonicalTrace]:
        """Validate trace constraints"""
        valid_traces = []
        
        for trace in traces:
            is_valid = True
            
            # Check certainty range
            if not (0.0 <= trace.certainty <= 1.0):
                print(f"⚠ Invalid certainty {trace.certainty} for trace {trace.trace_id}")
                trace.certainty = max(0.0, min(1.0, trace.certainty))
            
            # Check name not empty
            if not trace.name or len(trace.name) < 2:
                print(f"⚠ Invalid name for trace {trace.trace_id}")
                is_valid = False
            
            # FIXED: Updated valid focus list
            valid_focus = [
                "symptom", "sign", "lab", "imaging", "vital",  # MEDICAL FOCUSES
                "modifier", "radiation", "negation", 
                "temporal", "history", 
                "diagnosis", "constraint"
            ]
            if trace.focus not in valid_focus:
                print(f"⚠ Invalid focus '{trace.focus}' for trace {trace.trace_id}")
                is_valid = False
            
            if is_valid:
                valid_traces.append(trace)
        
        return valid_traces
    
    def normalize_all_traces(self, raw_traces: Dict) -> Dict[str, List[CanonicalTrace]]:
        """Main normalization pipeline"""
        
        print("🔄 Stage 2: Normalizing traces...")
        
        # Normalize each trace type
        evidence_canonical = self.normalize_evidence_traces(raw_traces["evidence"])
        hypothesis_canonical = self.normalize_hypothesis_traces(raw_traces["hypotheses"])
        counterfactual_canonical = self.normalize_counterfactual_traces(raw_traces["counterfactuals"])
        
        # Merge similar traces
        evidence_canonical = self._merge_similar_traces(evidence_canonical)
        
        # Validate all traces
        evidence_canonical = self.validate_traces(evidence_canonical)
        hypothesis_canonical = self.validate_traces(hypothesis_canonical)
        counterfactual_canonical = self.validate_traces(counterfactual_canonical)
        
        # Summary
        print(f"   ✓ Normalized {len(evidence_canonical)} evidence traces")
        print(f"   ✓ Normalized {len(hypothesis_canonical)} hypothesis traces")
        print(f"   ✓ Normalized {len(counterfactual_canonical)} counterfactual traces")
        
        total = len(evidence_canonical) + len(hypothesis_canonical) + len(counterfactual_canonical)
        print(f"   📊 Total canonical traces: {total}")
        
        return {
            "evidence": evidence_canonical,
            "hypotheses": hypothesis_canonical,
            "counterfactuals": counterfactual_canonical
        }
    
    def export_canonical_traces(self, canonical_traces: Dict) -> List[Dict]:
        """Export to flat list for downstream stages"""
        all_traces = []
        
        for trace_type, traces in canonical_traces.items():
            for trace in traces:
                all_traces.append({
                    "trace_id": trace.trace_id,
                    "trace_type": trace.trace_type,
                    "focus": trace.focus,
                    "name": trace.name,
                    "attributes": trace.attributes,
                    "certainty": trace.certainty,
                    "is_negated": trace.is_negated,
                    "source_traces": trace.source_traces
                })
        
        return all_traces

def test_normalization():
    """Test trace normalization"""
    from stage1_traces import EvidenceTrace, HypothesisTrace, CounterfactualTrace
    
    # Mock raw traces with issues
    raw_traces = {
        "evidence": [
            EvidenceTrace(
                focus="symptom",
                name="  CHEST PAIN  ",  # Needs normalization
                attributes={"location": "  Central ", "quality": "TIGHT"},
                certainty=0.9,
                source_sentence="S1"
            ),
            EvidenceTrace(
                focus="symptom",
                name="chest pain",  # Duplicate
                attributes={"duration": "2 days"},
                certainty=0.85,
                source_sentence="S2"
            ),
            EvidenceTrace(
                focus="negation",
                name="fever",
                certainty=1.0
            )
        ],
        "hypotheses": [
            HypothesisTrace(
                hypothesis="Angina Pectoris",
                snomed_search_terms=["angina", "chest pain exertional"],
                supporting_evidence=["chest pain", "exertional"],
                confidence=0.85
            )
        ],
        "counterfactuals": [
            CounterfactualTrace(
                hypothesis="Angina pectoris",
                counterfactual_condition="If pain were SHARP and POSITIONAL",
                impact="contradicts",
                reasoning="Would suggest musculoskeletal cause"
            )
        ]
    }
    
    # Normalize
    normalizer = TraceNormalizer()
    canonical = normalizer.normalize_all_traces(raw_traces)
    
    print("\n=== CANONICAL EVIDENCE TRACES ===")
    for trace in canonical["evidence"]:
        print(f"\nTrace ID: {trace.trace_id}")
        print(f"  Type: {trace.trace_type}")
        print(f"  Focus: {trace.focus}")
        print(f"  Name: {trace.name}")
        print(f"  Attributes: {trace.attributes}")
        print(f"  Certainty: {trace.certainty}")
        print(f"  Is Negated: {trace.is_negated}")
    
    print("\n=== CANONICAL HYPOTHESIS TRACES ===")
    for trace in canonical["hypotheses"]:
        print(f"\nTrace ID: {trace.trace_id}")
        print(f"  Name: {trace.name}")
        print(f"  Certainty: {trace.certainty}")
    
    print("\n=== CANONICAL COUNTERFACTUAL TRACES ===")
    for trace in canonical["counterfactuals"]:
        print(f"\nTrace ID: {trace.trace_id}")
        print(f"  Name: {trace.name}")
        print(f"  Condition: {trace.attributes.get('condition', '')[:50]}...")

if __name__ == "__main__":
    test_normalization()