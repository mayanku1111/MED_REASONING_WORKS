
import streamlit as st
from pipeline_main import EnhancedTraceToSNOMEDPipeline
import json
import time
import os
from datetime import datetime

# Page config
st.set_page_config(
    page_title="Trace-to-SNOMED Clinical Reasoning",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stProgress .st-bo {
        background-color: #00c851;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stage-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .snomed-concept {
        background-color: #e8f4f8;
        padding: 1rem;
        border-left: 4px solid #2196F3;
        margin: 0.5rem 0;
        border-radius: 0.25rem;
    }
    .trace-item {
        background-color: #ffffff;
        padding: 0.5rem;
        margin: 0.25rem 0;
        border-radius: 0.25rem;
        border-left: 3px solid #ffc107;
        color: #000000;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None
if 'results' not in st.session_state:
    st.session_state.results = None
if 'history' not in st.session_state:
    st.session_state.history = []

# Sidebar
with st.sidebar:
    st.title("⚙️ Configuration")
    
    api_key = st.text_input(
        "OpenRouter API Key",
        type="password",
        value=os.getenv("OPENROUTER_API_KEY", ""),
        help="Your OpenRouter API key"
    )
    
    snomed_url = st.text_input(
        "SNOMED Server URL",
        value="http://localhost:8080",
        help="Local SNOMED Snowstorm instance"
    )
    
    model = st.selectbox(
        "LLM Model",
        [
            "google/gemini-2.5-flash-lite",
            "deepseek/deepseek-r1-0528:free",
            "openai/gpt-oss-120b:free"
        ]
    )
    
    # NEW: Stage 2 normalization option
    use_normalization = st.checkbox(
        "Enable Stage 2 Normalization",
        value=True,
        help="Normalizes traces (deduplication, text cleanup, unique IDs). Recommended for production."
    )
    
    st.divider()
    
    if st.button("🔄 Initialize Pipeline", type="primary"):
        if api_key:
            try:
                st.session_state.pipeline = EnhancedTraceToSNOMEDPipeline(
                    openrouter_api_key=api_key,
                    snomed_base_url=snomed_url,
                    model=model,
                    use_normalization=use_normalization  # NEW
                )
                st.success("✅ Pipeline initialized!")
                if use_normalization:
                    st.info("Stage 2 normalization enabled")
                else:
                    st.info("Stage 2 normalization disabled (faster)")
            except Exception as e:
                st.error(f"❌ Error: {e}")
        else:
            st.error("Please provide API key")
    
    st.divider()
    
    # Pipeline status
    if st.session_state.pipeline:
        st.success("🟢 Pipeline Ready")
    else:
        st.warning("🟡 Pipeline Not Initialized")
    
    st.divider()
    
    # History
    st.subheader("📜 Query History")
    if st.session_state.history:
        for i, item in enumerate(reversed(st.session_state.history[-5:])):
            if st.button(f"Query {len(st.session_state.history)-i}", key=f"hist_{i}"):
                st.session_state.results = item

# Main content
st.title("Trace-to-SNOMED Clinical Reasoning System")
st.markdown("*Multi-agent reasoning with structured constraint satisfaction*")

# Example queries
with st.expander("📋 Example Complex Queries"):
    examples = {
        "Acute MI": """58-year-old male smoker with history of hypertension presents with 
severe crushing substernal chest pain for 2 hours. Pain radiates to left arm and jaw. 
Diaphoretic, 9/10 pain intensity. Started at rest, not relieved by position. 
Associated with nausea and dyspnea. No fever, no cough. Took aspirin without relief. 
Father died of heart attack at age 60.""",
        
        "Pneumonia": """32-year-old woman with fever (102°F), productive cough with 
yellow-green sputum for 4 days. Shortness of breath on exertion. 
Right-sided pleuritic chest pain. Recent upper respiratory infection. 
No known sick contacts. Non-smoker. No chronic conditions.""",
        
        "Pulmonary Embolism": """45-year-old woman with sudden onset severe dyspnea 
and right-sided chest pain 3 hours ago. Pain worsens with deep breathing. 
Recent 8-hour flight from Europe. History of oral contraceptive use. 
Mild cough. No fever. No leg swelling but reports left calf tenderness. 
Tachycardic (HR 115). Anxious appearing.""",
        
        "Diabetic Ketoacidosis": """22-year-old type 1 diabetic presents with 
severe nausea, vomiting, and abdominal pain for 24 hours. Excessive thirst and urination. 
Fruity breath odor noted. Confused and lethargic. Ran out of insulin 2 days ago. 
Deep, rapid breathing. No fever. Weight loss over past week."""
    }
    
    cols = st.columns(len(examples))
    for col, (name, query) in zip(cols, examples.items()):
        with col:
            if st.button(name, use_container_width=True):
                st.session_state.example_query = query

# Query input
query = st.text_area(
    "🩺 Enter Clinical Query",
    value=st.session_state.get('example_query', ''),
    height=150,
    placeholder="Describe patient presentation, symptoms, history..."
)

# Run pipeline button
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    run_button = st.button("🚀 Run Pipeline", type="primary", use_container_width=True)
with col2:
    clear_button = st.button("🗑️ Clear", use_container_width=True)
with col3:
    if st.session_state.results:
        if st.button("💾 Save Results", use_container_width=True):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"results_{timestamp}.json"
            with open(filename, "w") as f:
                json.dump(st.session_state.results, f, indent=2)
            st.success(f"Saved to {filename}")

if clear_button:
    st.session_state.results = None
    st.session_state.example_query = ""
    st.rerun()

# Run pipeline
if run_button and query:
    if not st.session_state.pipeline:
        st.error("❌ Please initialize pipeline first!")
    else:
        try:
            with st.spinner("🔄 Running pipeline..."):
                results = st.session_state.pipeline.run(query)
                st.session_state.results = results
                st.session_state.history.append(results)
            st.success("✅ Pipeline completed!")
            st.rerun()
        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.exception(e)

# Display results
if st.session_state.results:
    results = st.session_state.results
    
    # Final Answer at top
    st.markdown("---")
    st.markdown("### 🎯 Final Result")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "SNOMED Concept",
            results['stages']['stage8_answer']['snomed_concept']['name'].split('(')[0],
            delta=f"Score: {results['stages']['stage8_answer']['snomed_concept']['score']:.2f}"
        )
    with col2:
        st.metric(
            "Concept ID",
            results['stages']['stage8_answer']['snomed_concept']['id']
        )
    with col3:
        st.metric(
            "Processing Time",
            f"{results['metadata']['elapsed_time_seconds']:.1f}s"
        )
    
    st.markdown("#### 💡 Clinical Explanation")
    st.info(results['stages']['stage8_answer']['explanation'])
    
    st.markdown("#### ⚕️ Recommendation")
    st.warning(results['stages']['stage8_answer']['recommendation'])
    
    # Detailed stages
    st.markdown("---")
    st.markdown("### 📊 Pipeline Stages (Detailed View)")
    
    # STAGE 0: Preprocessing
    with st.expander("📋 STAGE 0: PREPROCESSING", expanded=False):
        st.markdown("*Deterministic sentence splitting, negation & temporal detection*")
        
        for sent in results['stages']['stage0_preprocessing']['sentences']:
            badge_color = {
                'statement': '🔵',
                'negation': '🔴',
                'temporal': '🟡'
            }.get(sent['type'], '⚪')
            
            st.markdown(f"""
            <div class="trace-item">
                {badge_color} <b>{sent['type'].upper()}</b><br>
                {sent['text']}<br>
                {f"<i>⏰ {sent['temporal_info']}</i>" if sent['temporal_info'] else ""}
            </div>
            """, unsafe_allow_html=True)
    
    # STAGE 1: Traces
    with st.expander("🤖 STAGE 1: MULTI-AGENT TRACE GENERATION", expanded=False):
        tab1, tab2, tab3 = st.tabs(["Evidence Traces", "Hypotheses", "Counterfactuals"])
        
        with tab1:
            st.markdown("**Agent A: Evidence Extraction**")
            for trace in results['stages']['stage1_traces']['evidence']:
                st.markdown(f"""
                <div class="trace-item">
                    <b>{trace['name']}</b> [{trace['focus']}]<br>
                    Certainty: {trace['certainty']}<br>
                    Attributes: {trace['attributes']}
                </div>
                """, unsafe_allow_html=True)
        
        with tab2:
            st.markdown("**Agent B: Hypothesis Planning**")
            for hyp in results['stages']['stage1_traces']['hypotheses']:
                st.markdown(f"""
                <div class="snomed-concept">
                    <b>{hyp['hypothesis']}</b><br>
                    Confidence: {hyp['confidence']}<br>
                    SNOMED Search Terms: {', '.join(hyp['snomed_search_terms'])}<br>
                    Evidence: {', '.join(hyp['supporting_evidence'])}
                </div>
                """, unsafe_allow_html=True)
        
        with tab3:
            st.markdown("**Agent C: Counterfactual Generation**")
            for cf in results['stages']['stage1_traces']['counterfactuals']:
                impact_emoji = {'supports': '✅', 'contradicts': '❌', 'neutral': '➖'}.get(cf['impact'], '❓')
                st.markdown(f"""
                <div class="trace-item">
                    <b>{cf['hypothesis']}</b> {impact_emoji}<br>
                    Counterfactual: {cf['counterfactual_condition']}<br>
                    <i>{cf['reasoning']}</i>
                </div>
                """, unsafe_allow_html=True)

    # STAGE 2: Normalization 
    if use_normalization and 'stage2_normalization' in results['stages']:
        with st.expander("🔧 STAGE 2: TRACE NORMALIZATION", expanded=False):
            st.markdown("*Deduplication, text cleanup, unique trace IDs*")
            norm = results['stages']['stage2_normalization']
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Original Evidence", norm.get('original_evidence_count', 0))
            with col2:
                st.metric("Original Hypotheses", norm.get('original_hypothesis_count', 0))
            with col3:
                st.metric("Total Canonical", norm.get('total_canonical_traces', 0))
            
            if 'sample_normalized' in norm:
                st.markdown("**Sample Normalized Traces:**")
                for trace in norm['sample_normalized'][:5]:
                    st.code(f"ID: {trace.get('trace_id', 'N/A')}\nName: {trace.get('name', 'N/A')}\nType: {trace.get('trace_type', 'N/A')}")
    
    
    # STAGE 3: Retrieval
    with st.expander("🔎 STAGE 3: SNOMED CT RETRIEVAL", expanded=False):
        st.metric("Total Candidates Retrieved", results['stages']['stage3_retrieval']['total_candidates'])
        
        for trace_key, candidates in results['stages']['stage3_retrieval']['by_trace'].items():
            with st.container():
                st.markdown(f"**{trace_key}**")
                for i, cand in enumerate(candidates[:5], 1):
                    strategy_badge = {
                        'lexical': '📝',
                        'lexical_enriched': '🧠',
                        'ontology': '🌳'
                    }.get(cand['strategy'], '❓')
                    st.markdown(f"{i}. {strategy_badge} `{cand['id']}` - {cand['name']}")
    
    # STAGE 4-5: Scoring
    with st.expander("📊 STAGE 4-5: SCORING & AGGREGATION", expanded=True):
        st.markdown("**Top 10 Scored Concepts**")
        
        for i, concept in enumerate(results['stages']['stage4_scoring']['scored_concepts'][:10], 1):
            with st.container():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**{i}. {concept['fsn']}**")
                    st.caption(f"ID: {concept['concept_id']}")
                with col2:
                    st.metric("Score", f"{concept['total_score']:.2f}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Supporting:**")
                    for trace in concept['supporting_traces']:
                        st.markdown(f"✅ {trace}")
                with col2:
                    if concept['contradicting_traces']:
                        st.markdown("**Contradicting:**")
                        for trace in concept['contradicting_traces']:
                            st.markdown(f"❌ {trace}")
                
                # Replace expander with a toggle button
                show_breakdown = st.checkbox(
                    f"Show score breakdown for concept {i}", 
                    key=f"breakdown_{i}"
                )
                if show_breakdown:
                    st.markdown("**Score Breakdown:**")
                    for key, score in concept['score_breakdown'].items():
                        st.text(f"  {key}: {score:.2f}")
                
                st.markdown("---")

    
    # STAGE 6: Adjudication
    with st.expander("⚖️ STAGE 6: LLM ADJUDICATION", expanded=False):
        adj = results['stages']['stage6_adjudication']
        
        if adj['was_needed']:
            st.success(f"Adjudication was needed - Winner: {adj['winner']}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Vote Ratio", f"{adj['vote_ratio']:.1%}")
            with col2:
                st.metric("Votes", str(adj['votes']))
        else:
            st.info("Adjudication was not needed - score gap was sufficient")
    
    # STAGE 8: Answer (already shown at top)
    with st.expander("📝 STAGE 8: GROUNDED ANSWER - FULL PROVENANCE", expanded=False):
        st.json(results['stages']['stage8_answer']['provenance'])

else:
    # Welcome screen
    st.info("👈 Configure the pipeline in the sidebar, then enter a clinical query above")
    
    st.markdown("### 🔬 How It Works")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("**Stage 0-1**")
        st.markdown("Preprocessing & Multi-agent trace generation")
    with col2:
        st.markdown("**Stage 3**")
        st.markdown("Multi-strategy SNOMED retrieval")
    with col3:
        st.markdown("**Stage 4-6**")
        st.markdown("Scoring, aggregation & adjudication")
    with col4:
        st.markdown("**Stage 8**")
        st.markdown("Grounded answer generation")