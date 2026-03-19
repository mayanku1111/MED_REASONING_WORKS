import os
from stage3_retrieval import SNOMEDRetriever

def test_pinpoint_pupils():
    """Test specifically for pinpoint pupils"""
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    
    print("=" * 60)
    print("Testing Pinpoint Pupils Synonym Expansion")
    print("=" * 60)
    
    retriever = SNOMEDRetriever(
        base_url="http://localhost:8080",
        api_key=api_key
    )
    
    # Test 1: Get synonyms
    print("\n1️⃣ Testing synonym generation:")
    synonyms = retriever.get_medical_synonyms("pinpoint pupils")
    print(f"Result: {synonyms}\n")
    
    # Test 2: Search with synonyms
    print("\n2️⃣ Testing search with synonyms:")
    results = retriever.lexical_search_with_synonyms("pinpoint pupils", limit=10)
    
    print(f"\n✅ Found {len(results)} concepts:")
    for i, r in enumerate(results[:5], 1):
        print(f"   {i}. {r.fsn}")
        print(f"      ID: {r.concept_id}")
        print(f"      Strategy: {r.retrieval_strategy}")
        print(f"      Matched: {r.matched_term}\n")

if __name__ == "__main__":
    test_pinpoint_pupils()