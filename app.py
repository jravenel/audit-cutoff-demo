#!/usr/bin/env python3
"""
Audit Cutoff Demo - Streamlit Application
RAG (LLM+Vector) vs Ontology Reasoning (SHACL/RDF) for Revenue Cut-off Audit
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rdflib import Graph
from pyshacl import validate
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Try to import advanced models, fallback to simple TF-IDF if not available
try:
    import faiss
    from sentence_transformers import SentenceTransformer
    ADVANCED_RAG_AVAILABLE = True
except ImportError:
    ADVANCED_RAG_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Audit Cutoff Demo",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'rag_model' not in st.session_state:
    st.session_state.rag_model = None
if 'rag_index' not in st.session_state:
    st.session_state.rag_index = None
if 'rag_metadata' not in st.session_state:
    st.session_state.rag_metadata = None

class RAGSearcher:
    """RAG search functionality - supports both advanced and simple modes"""
    
    def __init__(self):
        self.advanced_mode = False
        self.simple_mode = False
        self.model = None
        self.index = None
        self.metadata = None
        self.vectorizer = None
        self.tfidf_matrix = None
        
    def load_index(self, index_dir="rag/index"):
        """Load the RAG index - try advanced first, fallback to simple"""
        
        # Try advanced mode first
        if ADVANCED_RAG_AVAILABLE:
            try:
                # Load model
                model_path = os.path.join(index_dir, "model_name.txt")
                if os.path.exists(model_path):
                    with open(model_path, 'r') as f:
                        model_name = f.read().strip()
                else:
                    model_name = "all-MiniLM-L6-v2"
                
                if st.session_state.rag_model is None:
                    st.session_state.rag_model = SentenceTransformer(model_name)
                self.model = st.session_state.rag_model
                
                # Load FAISS index
                index_path = os.path.join(index_dir, "faiss_index.idx")
                if st.session_state.rag_index is None and os.path.exists(index_path):
                    st.session_state.rag_index = faiss.read_index(index_path)
                self.index = st.session_state.rag_index
                
                # Load metadata
                metadata_path = os.path.join(index_dir, "metadata.pkl")
                if st.session_state.rag_metadata is None and os.path.exists(metadata_path):
                    with open(metadata_path, 'rb') as f:
                        st.session_state.rag_metadata = pickle.load(f)
                self.metadata = st.session_state.rag_metadata
                
                if self.model and self.index and self.metadata:
                    self.advanced_mode = True
                    return True
                    
            except Exception as e:
                st.warning(f"Advanced RAG not available: {e}")
        
        # Fallback to simple mode
        try:
            simple_index_path = os.path.join(index_dir, "simple_index.pkl")
            if os.path.exists(simple_index_path):
                with open(simple_index_path, 'rb') as f:
                    index_data = pickle.load(f)
                
                self.vectorizer = index_data['vectorizer']
                self.tfidf_matrix = index_data['tfidf_matrix']
                self.metadata = index_data['metadata']
                self.simple_mode = True
                return True
                
        except Exception as e:
            st.error(f"Error loading simple RAG index: {e}")
        
        return False
    
    def search(self, query, k=5):
        """Search for relevant documents"""
        if self.advanced_mode:
            return self._search_advanced(query, k)
        elif self.simple_mode:
            return self._search_simple(query, k)
        else:
            return []
    
    def _search_advanced(self, query, k):
        """Advanced search using sentence transformers and FAISS"""
        try:
            # Encode query
            query_embedding = self.model.encode([query])
            import faiss
            faiss.normalize_L2(query_embedding)
            
            # Search
            scores, indices = self.index.search(query_embedding.astype('float32'), k)
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.metadata):
                    doc = self.metadata[idx]
                    results.append({
                        'score': float(score),
                        'content': doc['content'],
                        'type': doc['type'],
                        'source': doc['source'],
                        'metadata': doc
                    })
            
            return results
            
        except Exception as e:
            st.error(f"Error in advanced RAG search: {e}")
            return []
    
    def _search_simple(self, query, k):
        """Simple search using TF-IDF and cosine similarity"""
        try:
            # Transform query using the same vectorizer
            query_vector = self.vectorizer.transform([query])
            
            # Calculate cosine similarity
            similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
            
            # Get top k results
            top_indices = similarities.argsort()[-k:][::-1]
            
            results = []
            for idx in top_indices:
                if idx < len(self.metadata):
                    doc = self.metadata[idx]
                    score = similarities[idx]
                    results.append({
                        'score': float(score),
                        'content': doc['content'],
                        'type': doc['type'],
                        'source': doc['source'],
                        'metadata': doc
                    })
            
            return results
            
        except Exception as e:
            st.error(f"Error in simple RAG search: {e}")
            return []

class SHACLValidator:
    """SHACL validation functionality"""
    
    def __init__(self):
        self.data_graph = None
        self.shapes_graph = None
        
    def load_graphs(self):
        """Load RDF data and SHACL shapes"""
        try:
            # Load data graph
            data_path = "data/audit_data.ttl"
            if os.path.exists(data_path):
                self.data_graph = Graph()
                self.data_graph.parse(data_path, format="turtle")
            else:
                st.warning(f"RDF data file not found: {data_path}")
                return False
            
            # Load shapes graph
            shapes_path = "ont/shapes_cutoff.ttl"
            if os.path.exists(shapes_path):
                self.shapes_graph = Graph()
                self.shapes_graph.parse(shapes_path, format="turtle")
            else:
                st.warning(f"SHACL shapes file not found: {shapes_path}")
                return False
            
            return True
            
        except Exception as e:
            st.error(f"Error loading RDF graphs: {e}")
            return False
    
    def validate(self):
        """Run SHACL validation"""
        if self.data_graph is None or self.shapes_graph is None:
            return None, None, None
        
        try:
            # Run SHACL validation
            conforms, results_graph, results_text = validate(
                data_graph=self.data_graph,
                shacl_graph=self.shapes_graph,
                inference='rdfs',
                abort_on_error=False,
                meta_shacl=False,
                debug=False
            )
            
            return conforms, results_graph, results_text
            
        except Exception as e:
            st.error(f"Error in SHACL validation: {e}")
            return None, None, str(e)

def render_sidebar():
    """Render the sidebar with information"""
    st.sidebar.title("🔍 Audit Cutoff Demo")
    st.sidebar.markdown("""
    This demo compares two approaches for revenue cutoff audit:
    
    **📊 RAG (Retrieval-Augmented Generation)**
    - Vector search using sentence transformers
    - Semantic similarity matching
    - Natural language queries
    
    **🔗 Ontology Reasoning (SHACL/RDF)**
    - Formal logical validation
    - Rule-based constraint checking
    - Structured knowledge representation
    """)
    
    st.sidebar.markdown("---")
    
    # Data overview
    st.sidebar.subheader("📋 Data Overview")
    
    try:
        orders_df = pd.read_csv('data/orders.csv')
        deliveries_df = pd.read_csv('data/deliveries.csv')
        invoices_df = pd.read_csv('data/invoices.csv')
        je_df = pd.read_csv('data/journal_entries.csv')
        
        st.sidebar.metric("Orders", len(orders_df))
        st.sidebar.metric("Deliveries", len(deliveries_df))
        st.sidebar.metric("Invoices", len(invoices_df))
        st.sidebar.metric("Journal Entries", len(je_df))
        
    except Exception as e:
        st.sidebar.error("Error loading data files")

def render_rag_tab():
    """Render the RAG search tab"""
    st.header("📊 RAG Search: Semantic Audit Analysis")
    
    st.markdown("""
    Use natural language to search through audit policies and transaction data. 
    The system uses text similarity to find relevant content.
    """)
    
    # Initialize RAG searcher
    rag_searcher = RAGSearcher()
    
    # Check if any index exists
    simple_index_exists = os.path.exists("rag/index/simple_index.pkl")
    advanced_index_exists = os.path.exists("rag/index/faiss_index.idx")
    
    if not simple_index_exists and not advanced_index_exists:
        st.warning("⚠️ RAG index not found. Please run the setup first:")
        st.code("python rag/build_simple_index.py", language="bash")
        if st.button("🔧 Build Simple Index Now"):
            with st.spinner("Building simple RAG index..."):
                os.system("cd . && python rag/build_simple_index.py")
            st.rerun()
        return
    
    # Load index
    with st.spinner("Loading RAG index..."):
        if not rag_searcher.load_index():
            st.error("Failed to load RAG index")
            return
    
    # Show which mode is active
    if rag_searcher.advanced_mode:
        st.success("✅ Advanced RAG index loaded (sentence-transformers + FAISS)")
    elif rag_searcher.simple_mode:
        st.success("✅ Simple RAG index loaded (TF-IDF + cosine similarity)")
    
    # Search interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_input(
            "Enter your audit question:",
            placeholder="e.g., 'When should revenue be recognized?' or 'Show me cutoff violations'"
        )
    
    with col2:
        num_results = st.selectbox("Results", [3, 5, 10], index=1)
    
    # Predefined example queries
    st.markdown("**Example queries:**")
    example_queries = [
        "Revenue cutoff policy requirements",
        "When should revenue be recognized?",
        "Show me transactions with delivery issues",
        "Period end cutoff violations",
        "Revenue recognition before delivery",
        "Cutoff violation"
    ]
    
    cols = st.columns(len(example_queries))
    for i, example in enumerate(example_queries):
        if cols[i].button(f"💡 {example}", key=f"example_{i}"):
            query = example
            st.rerun()
    
    # Perform search
    if query:
        with st.spinner("Searching..."):
            results = rag_searcher.search(query, k=num_results)
        
        if results:
            st.subheader(f"🔍 Search Results for: '{query}'")
            
            for i, result in enumerate(results):
                with st.expander(f"Result {i+1} - {result['type'].title()} (Score: {result['score']:.3f})"):
                    st.markdown(f"**Source:** {result['source']}")
                    st.markdown(f"**Type:** {result['type']}")
                    st.markdown(f"**Relevance Score:** {result['score']:.4f}")
                    st.markdown("**Content:**")
                    st.write(result['content'])
                    
                    # Show additional metadata for transactions
                    if result['type'] in ['transaction', 'order']:
                        with st.expander("📋 Additional Details"):
                            st.json(result['metadata'])
        else:
            st.info("No results found. Try a different query.")

def render_shacl_tab():
    """Render the SHACL validation tab"""
    st.header("🔗 SHACL Validation: Formal Constraint Checking")
    
    st.markdown("""
    Formal validation using SHACL (Shapes Constraint Language) rules to detect 
    revenue cutoff violations based on structured ontology reasoning.
    """)
    
    # Initialize SHACL validator
    validator = SHACLValidator()
    
    # Check if RDF data exists
    if not os.path.exists("data/audit_data.ttl"):
        st.warning("⚠️ RDF data not found. Please run the setup first:")
        st.code("make setup", language="bash")
        return
    
    # Load graphs
    with st.spinner("Loading RDF data and SHACL shapes..."):
        if not validator.load_graphs():
            st.error("Failed to load RDF graphs")
            return
    
    st.success("✅ RDF graphs loaded successfully")
    
    # Show data statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Data Triples", len(validator.data_graph))
    with col2:
        st.metric("Shape Triples", len(validator.shapes_graph))
    with col3:
        if st.button("🔍 Run SHACL Validation", type="primary"):
            st.session_state.run_validation = True
    
    # Run validation
    if st.session_state.get('run_validation', False):
        with st.spinner("Running SHACL validation..."):
            conforms, results_graph, results_text = validator.validate()
        
        if conforms is None:
            st.error("Validation failed")
            return
        
        # Display results
        if conforms:
            st.success("✅ **VALIDATION PASSED**: No constraint violations found")
        else:
            st.error("❌ **VALIDATION FAILED**: Constraint violations detected")
        
        # Show validation results
        if results_text:
            st.subheader("📋 Validation Report")
            
            # Parse results to extract violations
            lines = results_text.split('\n')
            violations = []
            current_violation = {}
            
            for line in lines:
                line = line.strip()
                if line.startswith('Constraint Violation'):
                    if current_violation:
                        violations.append(current_violation)
                    current_violation = {'type': 'violation'}
                elif line.startswith('Message:'):
                    current_violation['message'] = line.replace('Message:', '').strip()
                elif line.startswith('Focus Node:'):
                    current_violation['focus_node'] = line.replace('Focus Node:', '').strip()
                elif line.startswith('Severity:'):
                    current_violation['severity'] = line.replace('Severity:', '').strip()
            
            if current_violation:
                violations.append(current_violation)
            
            if violations:
                st.subheader(f"🚨 Found {len(violations)} violation(s):")
                
                for i, violation in enumerate(violations):
                    with st.expander(f"Violation {i+1}: {violation.get('message', 'Unknown')[:100]}..."):
                        st.markdown(f"**Message:** {violation.get('message', 'N/A')}")
                        st.markdown(f"**Focus Node:** {violation.get('focus_node', 'N/A')}")
                        st.markdown(f"**Severity:** {violation.get('severity', 'N/A')}")
            
            # Show raw results for debugging
            with st.expander("🔧 Raw Validation Results"):
                st.text(results_text)
        
        # Visualization of the data
        st.subheader("📊 Transaction Timeline Analysis")
        
        try:
            # Load and visualize transaction data
            je_df = pd.read_csv('data/journal_entries.csv')
            orders_df = pd.read_csv('data/orders.csv')
            deliveries_df = pd.read_csv('data/deliveries.csv')
            
            # Create timeline visualization
            fig = go.Figure()
            
            # Add revenue transactions
            revenue_transactions = je_df[je_df['account_code'] == '4000']
            
            for _, row in revenue_transactions.iterrows():
                # Extract order ID from description
                order_id = None
                if 'ORD' in row['description']:
                    import re
                    match = re.search(r'ORD\d+', row['description'])
                    if match:
                        order_id = match.group()
                
                color = 'red' if 'VIOLATION' in row['description'] else 'green'
                
                fig.add_trace(go.Scatter(
                    x=[row['transaction_date']],
                    y=[row['credit_amount']],
                    mode='markers',
                    marker=dict(size=10, color=color),
                    name=f"Revenue: {order_id or row['journal_entry_id']}",
                    text=row['description'],
                    hovertemplate="<b>%{text}</b><br>Date: %{x}<br>Amount: $%{y:.2f}<extra></extra>"
                ))
            
            fig.update_layout(
                title="Revenue Recognition Timeline",
                xaxis_title="Transaction Date",
                yaxis_title="Revenue Amount ($)",
                hovermode='closest'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error creating visualization: {e}")

def main():
    """Main application"""
    st.title("🔍 Audit Cutoff Demo: RAG vs Ontology Reasoning")
    
    st.markdown("""
    This demo compares **RAG (Retrieval-Augmented Generation)** with **Ontology Reasoning (SHACL/RDF)** 
    for detecting revenue cutoff violations in financial audits.
    """)
    
    # Render sidebar
    render_sidebar()
    
    # Main tabs
    tab1, tab2 = st.tabs(["📊 RAG Search", "🔗 SHACL Validation"])
    
    with tab1:
        render_rag_tab()
    
    with tab2:
        render_shacl_tab()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **About this demo:** This application demonstrates two different approaches to audit automation:
    - **RAG**: Uses semantic search to find relevant policy text and transaction patterns
    - **SHACL**: Uses formal logical rules to validate data consistency and compliance
    
    Both approaches have their strengths and can be complementary in a comprehensive audit system.
    """)

if __name__ == "__main__":
    main()