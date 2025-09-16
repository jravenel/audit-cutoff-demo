# Audit Cutoff Demo: RAG vs Ontology Reasoning

A comprehensive demonstration comparing **RAG (Retrieval-Augmented Generation)** with **Ontology Reasoning (SHACL/RDF)** for revenue cut-off audit in financial systems.

## 🎯 Overview

This project showcases two different AI/reasoning approaches for detecting revenue cutoff violations:

1. **RAG Approach**: Uses vector search with sentence transformers (FAISS + sentence-transformers) for semantic similarity matching
2. **Ontology Approach**: Uses formal logical validation with SHACL (Shapes Constraint Language) and RDF knowledge graphs

## 🏗️ Architecture

```
audit-cutoff-demo/
├── app.py                      # Streamlit web application (2 tabs)
├── requirements.txt            # Python dependencies
├── Makefile                   # Easy setup and execution
├── csv_to_rdf.py             # ETL: Convert CSV data to RDF
├── data/                     # Sample data with cutoff violations
│   ├── orders.csv           # Customer orders
│   ├── deliveries.csv       # Delivery confirmations
│   ├── invoices.csv         # Invoice data
│   ├── journal_entries.csv  # Accounting entries (includes violations)
│   └── revenue_cutoff_policy.txt  # Audit policy text
├── ont/                      # Ontology definitions
│   ├── cutoff_ontology.ttl  # Domain ontology (OWL)
│   └── shapes_cutoff.ttl    # SHACL validation shapes
└── rag/                      # RAG system components
    └── build_index.py       # Build FAISS index with sentence-transformers
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip package manager

### Installation & Setup

1. **Clone and navigate to the repository:**
   ```bash
   git clone <repository-url>
   cd audit-cutoff-demo
   ```

2. **Install dependencies:**
   ```bash
   make install
   ```

3. **Set up the complete environment (ETL + RAG index):**
   ```bash
   make setup
   ```

4. **Run the application:**
   ```bash
   make run
   ```

5. **Open your browser** to the displayed URL (typically `http://localhost:8501`)

### Manual Setup (Alternative)

```bash
# Install dependencies
pip install -r requirements.txt

# Convert CSV to RDF
python csv_to_rdf.py

# Build RAG index
python rag/build_index.py

# Run the app
streamlit run app.py
```

## 📊 Demo Data

The demo includes realistic financial data with **one deliberate cutoff violation**:

- **Orders**: 7 sample customer orders across year-end (2023-2024)
- **Deliveries**: Corresponding delivery confirmations with dates
- **Invoices**: Invoice data with various timing scenarios
- **Journal Entries**: Accounting entries including one revenue cutoff violation (ORD004)

### The Cutoff Violation

**Order ORD004** demonstrates a classic revenue cutoff violation:
- **Order Date**: 2023-12-31
- **Delivery Date**: 2024-01-03 (3 days later)
- **Revenue Recognition**: 2023-12-31 ❌ (Should be 2024-01-03)

This violates the principle that revenue should only be recognized when goods are delivered to the customer.

## 🔍 Features

### RAG Search Tab
- **Semantic Search**: Natural language queries against policy text and transaction data
- **Vector Embeddings**: Uses sentence-transformers for semantic similarity
- **Interactive Examples**: Pre-built example queries
- **Relevance Scoring**: Shows similarity scores for search results

### SHACL Validation Tab
- **Formal Validation**: Rule-based constraint checking using SHACL
- **Violation Detection**: Automatically identifies cutoff violations
- **RDF Knowledge Graph**: Structured representation of audit data
- **Timeline Visualization**: Interactive chart showing transaction patterns

## 🧠 Technical Details

### RAG Implementation
- **Model**: `all-MiniLM-L6-v2` sentence transformer
- **Index**: FAISS with cosine similarity
- **Documents**: Policy text + transaction descriptions
- **Search**: Semantic similarity matching

### Ontology Implementation
- **Format**: RDF/Turtle with OWL ontology
- **Validation**: SHACL constraint shapes
- **Rules**: SPARQL-based cutoff validation logic
- **Engine**: pyshacl for validation execution

### Key SHACL Rules
1. **Revenue Cutoff**: Revenue cannot be recognized before delivery
2. **Delivery Required**: Revenue entries must have completed deliveries
3. **Period End Cutoff**: Year-end boundary validation
4. **Data Consistency**: Invoice-order relationship validation

## 📈 Comparison: RAG vs Ontology

| Aspect | RAG Approach | Ontology Approach |
|--------|-------------|-------------------|
| **Query Style** | Natural language | Formal logical rules |
| **Flexibility** | High - handles varied queries | Medium - requires rule definition |
| **Precision** | Semantic similarity based | Exact logical validation |
| **Setup Complexity** | Medium - requires embeddings | High - requires ontology design |
| **Explainability** | Similarity scores | Formal logical proofs |
| **Domain Knowledge** | Implicit in embeddings | Explicit in ontology |
| **False Positives** | Possible with similar text | Minimal with precise rules |
| **Scalability** | Good with proper indexing | Excellent with optimized reasoners |

## 🛠️ Available Commands

```bash
make install    # Install Python dependencies
make setup      # Full setup (ETL + RAG index)
make run        # Run Streamlit application
make clean      # Clean generated files
make test       # Test dependencies
make help       # Show available commands
```

## 📁 File Descriptions

### Core Application
- **`app.py`**: Main Streamlit application with two-tab interface
- **`requirements.txt`**: Python package dependencies

### Data Processing
- **`csv_to_rdf.py`**: ETL script converting CSV data to RDF triples
- **`data/*.csv`**: Sample financial transaction data
- **`data/revenue_cutoff_policy.txt`**: Audit policy documentation

### Ontology & Rules
- **`ont/cutoff_ontology.ttl`**: OWL ontology defining domain concepts
- **`ont/shapes_cutoff.ttl`**: SHACL shapes for validation rules

### RAG System
- **`rag/build_index.py`**: Builds FAISS vector index for semantic search

## 🎓 Educational Value

This demo illustrates:

1. **Semantic Search vs Rule-Based Validation**
2. **Vector Embeddings in Financial Audit**
3. **Knowledge Graphs for Compliance**
4. **SHACL for Data Quality Validation**
5. **Hybrid AI Approaches in Enterprise**

## 🤝 Use Cases

- **Financial Auditing**: Revenue recognition compliance
- **Compliance Monitoring**: Automated policy adherence
- **Educational**: Teaching AI reasoning approaches
- **Research**: Comparing semantic vs symbolic AI

## 📄 License

This project is for demonstration and educational purposes.

## 🔗 Technologies Used

- **Frontend**: Streamlit
- **RAG**: sentence-transformers, FAISS
- **Ontology**: RDFLib, pyshacl
- **Data**: Pandas, NumPy
- **Visualization**: Plotly

---

**Ready to explore?** Run `make setup && make run` to start the demo!
