import os
import streamlit as st
import chromadb
from groq import Groq
import json

os.environ["TOKENIZERS_PARALLELISM"] = "false"

GROQ_API_KEY = "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"
   # "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"
groq_client = Groq(api_key=GROQ_API_KEY)

st.set_page_config(
    page_title="Localmap-A RAG based tool to explore local places using Reddit",
    layout="wide"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        margin: 10px 0;
    }
    .metric-label {
        font-size: 14px;
        color: #6c757d;
        font-weight: 500;
        margin-bottom: 5px;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 600;
        color: #212529;
    }
    .metric-delta {
        font-size: 12px;
        color: #6c757d;
        margin-top: 5px;
    }
    .score-high {
        color: #28a745;
    }
    .score-medium {
        color: #ffc107;
    }
    .score-low {
        color: #dc3545;
    }
    .section-header {
        font-size: 18px;
        font-weight: 600;
        color: #343a40;
        margin: 20px 0 10px 0;
        padding-bottom: 10px;
        border-bottom: 2px solid #e9ecef;
    }
    .document-card {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 6px;
        border: 1px solid #e9ecef;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


class RAGEvaluator:
    """Lightweight RAG evaluator"""

    def __init__(self, groq_client):
        self.client = groq_client

    def _call_llm(self, system_prompt: str, user_prompt: str) -> dict:
        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except:
            return {}

    def evaluate_retrieval(self, question: str, documents: list) -> dict:
        """Evaluate retrieval relevance"""
        system = "Grade if documents are relevant. Respond in JSON: {relevant: bool, score: 1-5, explanation: str}"
        docs = "\n".join([f"Doc{i + 1}: {d[:150]}" for i, d in enumerate(documents)])
        user = f"Q: {question}\n\nDocs:\n{docs}"
        return self._call_llm(system, user)

    def evaluate_answer(self, question: str, documents: list, answer: str) -> dict:
        """Evaluate answer quality"""
        system = "Grade answer quality. JSON: {is_grounded: bool, is_helpful: bool, has_hallucination: bool, quality_score: 1-5, explanation: str}"
        docs = "\n".join([f"Fact{i + 1}: {d[:120]}" for i, d in enumerate(documents)])
        user = f"Q: {question}\n\nFacts:\n{docs}\n\nAnswer: {answer}"
        return self._call_llm(system, user)

    def evaluate_all(self, question: str, documents: list, answer: str) -> dict:
        """Run all evaluations"""
        retrieval = self.evaluate_retrieval(question, documents)
        quality = self.evaluate_answer(question, documents, answer)

        return {
            'retrieval': retrieval,
            'quality': quality,
            'overall_score': (retrieval.get('score', 1) + quality.get('quality_score', 1)) / 2
        }


class PlacesApp:
    def __init__(self):
        self.client = None
        self.collection = None
        self.evaluator = RAGEvaluator(groq_client)

    def connect_to_db(self):
        if self.client is None:
            self.client = chromadb.CloudClient(
                api_key="ck-Bz6vWE8LH4mrmKx2dfreGSzgiikFBKhm7BHNHivkDvvw",
                tenant="6db89e03-6466-4af4-ad1c-d8237a75efa7",
                database="places",
            )
        self.collection = self.client.get_collection(name="cleaned_posts")
        return True

    def search_and_answer(self, query: str, k: int = 8, evaluate: bool = False):
        """RAG pipeline with optional evaluation"""

        # Retrieve documents
        results = self.collection.query(
            query_texts=[query],
            n_results=k,
        )

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]

        # Build context
        context_parts = []
        for doc, meta in zip(documents, metadatas):
            place = meta.get("places", meta.get("place", "N/A"))
            city = meta.get("city", "Unknown")
            location = meta.get("location", "N/A")
            theme = meta.get("theme", "N/A")

            context_parts.append(
                f"Place: {place} | City: {city} | Location: {location} | "
                f"Theme: {theme} | Notes: {doc[:200]}"
            )

        context = "\n\n".join(context_parts)

        # Generate answer
        completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system",
                 "content": "You are a helpful travel assistant. Recommend places based ONLY on provided context."},
                {"role": "user", "content": f"Question: {query}\n\nContext:\n{context}\n\nProvide recommendations:"}
            ],
            temperature=0.3
        )

        answer = completion.choices[0].message.content

        # Evaluate if requested
        evaluation = None
        if evaluate:
            evaluation = self.evaluator.evaluate_all(query, documents, answer)

        return {
            'answer': answer,
            'documents': documents,
            'metadatas': metadatas,
            'evaluation': evaluation
        }


# Initialize session state
if "app" not in st.session_state:
    st.session_state.app = PlacesApp()
    st.session_state.connected = False
    st.session_state.results = None

app = st.session_state.app

# Header
st.title("Localmap-A RAG based tool to explore local places using Reddit")
st.markdown("Search for places and view retrieval quality metrics")

# Sidebar
with st.sidebar:
    st.header("Settings")

    if not st.session_state.connected:
        if st.button("Connect to Database", type="primary", use_container_width=True):
            with st.spinner("Connecting to database..."):
                try:
                    app.connect_to_db()
                    st.session_state.connected = True
                    st.rerun()
                except Exception as e:
                    st.error(f"Connection failed: {e}")
    else:
        st.success("Database Connected")

    st.divider()

    st.subheader("Evaluation Options")
    enable_eval = st.checkbox("Enable RAG Evaluation", value=True,
                              help="Evaluate retrieval and answer quality using LLM")

    st.subheader("Search Parameters")
    num_results = st.slider("Number of results:", 5, 15, 8)

    st.divider()
    st.caption("Evaluation uses LLM to grade retrieval relevance and answer quality")

# Main content
if not st.session_state.connected:
    st.warning("Please connect to the database to begin")
else:
    # Search section
    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_input("Search Query", placeholder="e.g., best pizza in New York", label_visibility="collapsed")
    with col2:
        search_btn = st.button("Search", type="primary", use_container_width=True)

    if search_btn and query:
        with st.spinner("Processing query..."):
            try:
                results = app.search_and_answer(query, k=num_results, evaluate=enable_eval)
                st.session_state.results = results
            except Exception as e:
                st.error(f"Error: {e}")

    # Display results
    if st.session_state.results:
        results = st.session_state.results

        # Evaluation metrics
        if enable_eval and results.get('evaluation'):
            st.divider()
            st.markdown('<div class="section-header">RAG Evaluation Metrics</div>', unsafe_allow_html=True)

            eval_data = results['evaluation']

            col1, col2, col3 = st.columns(3)

            with col1:
                retrieval = eval_data['retrieval']
                score = retrieval.get('score', 0)
                relevant = retrieval.get('relevant', False)

                score_class = "score-high" if score >= 4 else "score-medium" if score >= 3 else "score-low"

                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Retrieval Relevance</div>
                    <div class="metric-value {score_class}">{score}/5</div>
                    <div class="metric-delta">{"Relevant" if relevant else "Not Relevant"}</div>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                quality = eval_data['quality']
                q_score = quality.get('quality_score', 0)
                grounded = quality.get('is_grounded', False)

                score_class = "score-high" if q_score >= 4 else "score-medium" if q_score >= 3 else "score-low"

                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Answer Quality</div>
                    <div class="metric-value {score_class}">{q_score}/5</div>
                    <div class="metric-delta">{"Grounded" if grounded else "Not Grounded"}</div>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                overall = eval_data.get('overall_score', 0)
                score_class = "score-high" if overall >= 4 else "score-medium" if overall >= 3 else "score-low"

                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Overall Score</div>
                    <div class="metric-value {score_class}">{overall:.1f}/5</div>
                    <div class="metric-delta">Average</div>
                </div>
                """, unsafe_allow_html=True)

            # Detailed evaluation
            with st.expander("View Detailed Evaluation"):
                st.markdown("**Retrieval Analysis:**")
                st.write(retrieval.get('explanation', 'N/A'))

                st.markdown("**Answer Quality Analysis:**")
                st.write(quality.get('explanation', 'N/A'))

                hallucination = quality.get('has_hallucination', False)
                if hallucination:
                    st.warning("Warning: Possible hallucination detected in answer")

                helpful = quality.get('is_helpful', False)
                if helpful:
                    st.info("Answer is helpful and addresses the query")

        # Answer section
        st.divider()
        st.markdown('<div class="section-header">Recommendations</div>', unsafe_allow_html=True)
        st.write(results['answer'])

        # Documents section
        st.divider()
        st.markdown('<div class="section-header">Retrieved Documents</div>', unsafe_allow_html=True)

        for i, (doc, meta) in enumerate(zip(results['documents'], results['metadatas']), 1):
            with st.expander(f"Document {i}: {meta.get('places', 'Unknown')}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**City:** {meta.get('city', 'N/A')}")
                    st.markdown(f"**Location:** {meta.get('location', 'N/A')}")
                with col2:
                    st.markdown(f"**Theme:** {meta.get('theme', 'N/A')}")
                    st.markdown(f"**Sentiment:** {meta.get('sentiment', 'N/A')}")
                st.divider()
                st.text_area("Content", doc[:500], height=100, disabled=True, label_visibility="collapsed")

st.divider()
st.caption("Powered by ChromaDB, Groq, and Streamlit with integrated RAG evaluation")