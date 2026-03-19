import requests
import json
from typing import List, Dict, Set
from dataclasses import dataclass
from openrouter import OpenRouter

@dataclass
class SNOMEDConcept:
    """Structured SNOMED concept"""
    concept_id: str
    fsn: str                      # Fully Specified Name
    pt: str = None                # Preferred Term
    definition: str = None
    retrieval_strategy: str = "lexical"   # lexical | lexical_enriched | ontology | synonym
    matched_term: str = None      # Which term actually matched


# =========================
# Retriever
# =========================

class SNOMEDRetriever:
    """Stage 3: Multi-strategy SNOMED retrieval with LLM synonym expansion"""

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        branch: str = "MAIN",
        api_key: str = None,
        model: str = "google/gemini-2.5-flash-lite"
    ):
        self.base_url = base_url
        self.branch = branch
        self.api_key = api_key
        self.model = model

        self.headers = {
            "Accept": "application/json",
            "User-Agent": "TraceToSNOMED/1.0"
        }

        self.synonym_cache: Dict[str, List[str]] = {}
        self.search_cache: Dict[str, List[SNOMEDConcept]] = {}

    # =========================
    # Internal API Helper
    # =========================

    def _api_call(self, endpoint: str, params: Dict) -> Dict:
        try:
            url = f"{self.base_url}/{self.branch}/{endpoint}"
            r = requests.get(url, params=params, headers=self.headers, timeout=10)
            if r.status_code == 200:
                return r.json()
            print(f"⚠ API Error {r.status_code}: {r.text[:120]}")
        except Exception as e:
            print(f"⚠ Connection error: {e}")
        return {"items": []}

    # LLM Synonym Expansion


    def get_medical_synonyms(self, term: str, max_synonyms: int = 5) -> List[str]:
        """
        Use LLM to generate SNOMED-compatible medical synonyms.
        Always returns a list including the original term.
        """

        term_clean = term.strip()
        term_lower = term_clean.lower()

        # Cache
        if term_lower in self.synonym_cache:
            print(f"      📦 Using cached synonyms for '{term_clean}'")
            return self.synonym_cache[term_lower]

        # No API key fallback
        if not self.api_key:
            print(f"      ⚠ No API key, skipping synonym expansion")
            return [term_clean]

        prompt = f"""
You are a medical terminology expert specializing in SNOMED CT.

For the clinical term "{term_clean}", generate {max_synonyms} SNOMED-compatible synonyms.

CRITICAL RULES:
1. Include the FORMAL medical term (ontology-preferred)
2. Include common clinical synonyms
3. Include anatomical/pathological variants
4. DO NOT include vague or lay terms
5. Prefer terms that appear in medical ontologies

EXAMPLES:

Input: "pinpoint pupils"
Output: {{"synonyms": ["miosis", "bilateral miosis", "constricted pupil", "pupillary constriction", "small pupils"]}}

Input: "fast heart rate"
Output: {{"synonyms": ["tachycardia", "sinus tachycardia", "rapid heart rate", "elevated heart rate", "increased pulse rate"]}}

Now generate for: "{term_clean}"

Return ONLY valid JSON:
{{"synonyms": ["term1", "term2", "term3", "term4", "term5"]}}
"""

        try:
            with OpenRouter(api_key=self.api_key) as client:
                response = client.chat.send(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a medical terminology expert. Always return valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.2,
                    reasoning={"enabled": False}
                )

            raw = response.choices[0].message.content
            print(f"      🔍 LLM response: {raw[:160]}...")

            data = json.loads(raw)
            synonyms = data.get("synonyms", [])

            if not isinstance(synonyms, list):
                raise ValueError("Invalid synonym format")

            # Normalize + dedupe
            final_terms = [term_clean]
            for s in synonyms:
                if isinstance(s, str) and s.lower() != term_lower:
                    final_terms.append(s.strip())

            final_terms = final_terms[: max_synonyms + 1]

            self.synonym_cache[term_lower] = final_terms
            print(f"      ✅ Expanded '{term_clean}' → {final_terms}")

            return final_terms

        except Exception as e:
            print(f"      ⚠ Synonym expansion failed: {str(e)[:80]}")
            return [term_clean]

    # =========================
    # Lexical Search
    # =========================

    def lexical_search(self, query: str, limit: int = 15) -> List[SNOMEDConcept]:
        params = {
            "term": query,
            "activeFilter": "true",
            "limit": limit
        }

        result = self._api_call("concepts", params)

        concepts = []
        for item in result.get("items", []):
            concepts.append(
                SNOMEDConcept(
                    concept_id=item["conceptId"],
                    fsn=item["fsn"]["term"],
                    pt=item.get("pt", {}).get("term"),
                    retrieval_strategy="lexical",
                    matched_term=query
                )
            )
        return concepts

    # =========================
    # Lexical Search + Synonyms
    # =========================

    def lexical_search_with_synonyms(self, query: str, limit: int = 15) -> List[SNOMEDConcept]:
        results = self.lexical_search(query, limit=limit)

        if len(results) >= 3:
            print(f"      ✓ Found {len(results)} via direct search")
            return results

        print(f"      ⚠ Only {len(results)} results, expanding with synonyms...")

        variants = self.get_medical_synonyms(query, max_synonyms=5)

        all_results = list(results)
        seen_ids = {c.concept_id for c in results}

        for i, variant in enumerate(variants[1:], 1):
            print(f"      → Variant {i}: '{variant}'")
            variant_results = self.lexical_search(variant, limit=10)

            for concept in variant_results:
                if concept.concept_id not in seen_ids:
                    concept.retrieval_strategy = "synonym"
                    concept.matched_term = variant
                    all_results.append(concept)
                    seen_ids.add(concept.concept_id)

        print(f"      ✅ Total candidates: {len(all_results)}")
        return all_results[:limit]

    # =========================
    # Enriched Search
    # =========================

    def lexical_search_enriched(self, trace_text: str, attributes: Dict, limit: int = 15):
        parts = [trace_text]

        for key in ["location", "quality", "pattern"]:
            if attributes.get(key):
                parts.append(str(attributes[key]))

        for k, v in attributes.items():
            if v and k not in ["location", "quality", "pattern"]:
                parts.append(str(v))

        query = " ".join(parts[:4])

        concepts = self.lexical_search_with_synonyms(query, limit)
        for c in concepts:
            if c.retrieval_strategy == "lexical":
                c.retrieval_strategy = "lexical_enriched"

        return concepts

    # =========================
    # Ontology Expansion
    # =========================

    def ontology_expansion(self, concept_id: str, limit: int = 20):
        result = self._api_call("concepts", {"ecl": f"< {concept_id}", "limit": limit})

        return [
            SNOMEDConcept(
                concept_id=i["conceptId"],
                fsn=i["fsn"]["term"],
                pt=i.get("pt", {}).get("term"),
                retrieval_strategy="ontology"
            )
            for i in result.get("items", [])
        ]

    def multi_strategy_retrieval(self, trace_name: str, trace_attributes: Dict = None) -> List[SNOMEDConcept]:
        """
        Combined retrieval strategy with synonym expansion
        
        Strategies:
        1. Basic Lexical Search (with synonym fallback)
        2. Enriched Lexical Search (with attributes + synonyms)
        3. Ontology Expansion
        """
        # Check cache first
        cache_key = f"{trace_name}:{bool(trace_attributes)}"
        if cache_key in self.search_cache:
            print(f"      📦 Using cached results")
            return self.search_cache[cache_key]
        
        all_candidates = []
        seen_ids: Set[str] = set()
        
        # Strategy 1: Basic Lexical Search (with synonym fallback)
        print(f"      → Lexical search: {trace_name}")
        lexical_results = self.lexical_search_with_synonyms(trace_name, limit=10)
        for concept in lexical_results:
            if concept.concept_id not in seen_ids:
                all_candidates.append(concept)
                seen_ids.add(concept.concept_id)
        
        # Strategy 2: Enriched Lexical Search (with attributes)
        if trace_attributes and any(trace_attributes.values()):
            print(f"      → Enriched search with attributes")
            enriched_results = self.lexical_search_enriched(trace_name, trace_attributes, limit=10)
            for concept in enriched_results:
                if concept.concept_id not in seen_ids:
                    all_candidates.append(concept)
                    seen_ids.add(concept.concept_id)
        
        # Strategy 3: Ontology Expansion (from top lexical result)
        if lexical_results:
            top_concept_id = lexical_results[0].concept_id
            print(f"      → Ontology expansion from: {lexical_results[0].fsn[:50]}...")
            ontology_results = self.ontology_expansion(top_concept_id, limit=15)
            for concept in ontology_results:
                if concept.concept_id not in seen_ids:
                    all_candidates.append(concept)
                    seen_ids.add(concept.concept_id)
        
        # Cache results
        self.search_cache[cache_key] = all_candidates
        
        return all_candidates

    def retrieve_for_traces(self, evidence_traces, hypotheses) -> Dict[str, List[SNOMEDConcept]]:
        """Retrieve candidates for all traces"""
        print("🔎 Stage 3: Retrieving SNOMED candidates...")
        
        # CLEAR CACHE at start of new pipeline run
        self.search_cache.clear()
        
        results = {}
        
        # Retrieve for evidence traces
        print("\n   📋 Evidence Traces:")
        for trace in evidence_traces:
            print(f"   Searching: {trace.name}")
            candidates = self.multi_strategy_retrieval(trace.name, trace.attributes)
            results[f"evidence:{trace.name}"] = candidates
            print(f"     ✓ Found {len(candidates)} candidates\n")
        
        # Retrieve for hypothesis search terms
        print("   💡 Hypothesis Search Terms:")
        for hyp in hypotheses:
            for search_term in hyp.snomed_search_terms[:2]:  # Limit to top 2 terms per hypothesis
                print(f"   Searching: {search_term} (from {hyp.hypothesis})")
                candidates = self.multi_strategy_retrieval(search_term)
                results[f"hypothesis:{search_term}"] = candidates
                print(f"     ✓ Found {len(candidates)} candidates\n")
        
        # Summary
        total_unique = len(set(c.concept_id for candidates in results.values() for c in candidates))
        print(f"   📊 Total unique SNOMED concepts retrieved: {total_unique}")
        
        return results