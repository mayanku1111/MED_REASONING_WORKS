
import os
import sys

def check_dependencies():
    """Check if all required packages are installed"""
    print("🔍 Checking dependencies...")
    
    required = ['openrouter', 'pydantic', 'requests', 'streamlit']
    missing = []
    
    for package in required:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} - MISSING")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("\n✅ All dependencies installed!")
    return True

def check_api_key():
    """Check if OpenRouter API key is set"""
    print("\n🔑 Checking API key...")
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    
    if not api_key:
        print("  ❌ OPENROUTER_API_KEY not found")
        print("\nSet your API key:")
        print("  export OPENROUTER_API_KEY='your_key_here'")
        print("  OR create a .env file with:")
        print("  OPENROUTER_API_KEY=your_key_here")
        return False
    
    print(f"  ✅ API key found ({api_key[:10]}...)")
    return True

def check_snomed():
    """Check if SNOMED Docker instance is accessible"""
    print("\n🏥 Checking SNOMED Snowstorm instance...")
    
    try:
        import requests
        response = requests.get("http://localhost:8080/version", timeout=5)
        
        if response.status_code == 200:
            version_info = response.json()
            print(f"  ✅ SNOMED Snowstorm connected")
            print(f"     Version: {version_info.get('version', 'unknown')}")
            return True
        else:
            print(f"  ❌ Server responded with status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"  ❌ Cannot connect to SNOMED server")
        print(f"     Error: {e}")
        print("\nMake sure your SNOMED Docker container is running:")
        print("  docker ps")
        print("  curl http://localhost:8080/version")
        return False

def run_simple_test():
    """Run a simple end-to-end test"""
    print("\n🧪 Running simple test...")
    
    try:
        from pipeline_main import TraceToSNOMEDPipeline
        
        # Simple test query
        test_query = "Patient with fever and cough for 3 days."
        
        print(f"\nTest query: {test_query}")
        print("\nInitializing pipeline...")
        
        pipeline = TraceToSNOMEDPipeline(
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY")
        )
        
        print("Running pipeline (this may take 30-60 seconds)...")
        result = pipeline.run(test_query)
        
        print("\n✅ TEST PASSED!")
        print(f"\nResult:")
        print(f"  Concept: {result['stages']['stage8_answer']['snomed_concept']['name']}")
        print(f"  ID: {result['stages']['stage8_answer']['snomed_concept']['id']}")
        print(f"  Time: {result['metadata']['elapsed_time_seconds']:.1f}s")
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main quickstart check"""
    print("="*60)
    print("🚀 TRACE-TO-SNOMED QUICKSTART CHECK")
    print("="*60)
    
    checks = [
        ("Dependencies", check_dependencies()),
        ("API Key", check_api_key()),
        ("SNOMED Server", check_snomed())
    ]
    
    all_passed = all(result for _, result in checks)
    
    if not all_passed:
        print("\n" + "="*60)
        print("❌ SETUP INCOMPLETE")
        print("="*60)
        print("\nFix the issues above before running the pipeline.")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("✅ ALL CHECKS PASSED!")
    print("="*60)
    
    # Ask if user wants to run test
    print("\nWould you like to run a simple end-to-end test? (y/n)")
    response = input("> ").strip().lower()
    
    if response == 'y':
        if run_simple_test():
            print("\n" + "="*60)
            print("🎉 SETUP COMPLETE!")
            print("="*60)
            print("\nYou're ready to go! Try:")
            print("  1. streamlit run streamlit_app.py  (for GUI)")
            print("  2. python pipeline_main.py  (for CLI)")
            print("  3. python test_complex_cases.py  (for testing)")
        else:
            print("\n" + "="*60)
            print("⚠️  Test failed - check error messages above")
            print("="*60)
    else:
        print("\n" + "="*60)
        print("✅ SETUP VERIFIED!")
        print("="*60)
        print("\nYou're ready! Run:")
        print("  streamlit run streamlit_app.py")

if __name__ == "__main__":
    main()