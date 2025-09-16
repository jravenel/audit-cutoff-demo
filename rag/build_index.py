#!/usr/bin/env python3
"""
RAG Index Builder: Build FAISS index for semantic search
Creates embeddings from audit policy text and transaction data for RAG search
"""

import os
import pandas as pd
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any

class RAGIndexBuilder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the RAG index builder with sentence transformer model"""
        print(f"Loading sentence transformer model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.documents = []
        self.metadata = []
        
    def load_policy_text(self) -> List[Dict[str, Any]]:
        """Load and chunk the revenue cutoff policy text"""
        policy_file = "data/revenue_cutoff_policy.txt"
        
        if not os.path.exists(policy_file):
            print(f"Warning: Policy file {policy_file} not found")
            return []
        
        with open(policy_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split into logical chunks (by sections)
        sections = content.split('\n## ')
        documents = []
        
        for i, section in enumerate(sections):
            if section.strip():
                # Clean up section headers
                if not section.startswith('##'):
                    section = '## ' + section
                
                documents.append({
                    'content': section.strip(),
                    'type': 'policy',
                    'section_id': i,
                    'source': 'revenue_cutoff_policy.txt'
                })
        
        return documents
    
    def load_transaction_examples(self) -> List[Dict[str, Any]]:
        """Load transaction data as examples for semantic search"""
        documents = []
        
        try:
            # Load journal entries
            je_df = pd.read_csv('data/journal_entries.csv')
            
            for _, row in je_df.iterrows():
                # Create descriptive text for each transaction
                content = f"Journal Entry {row['journal_entry_id']}: {row['description']} " \
                         f"Date: {row['transaction_date']}, " \
                         f"Account: {row['account_name']} ({row['account_code']}), " \
                         f"Debit: ${row['debit_amount']:.2f}, Credit: ${row['credit_amount']:.2f}"
                
                documents.append({
                    'content': content,
                    'type': 'transaction',
                    'transaction_id': row['journal_entry_id'],
                    'date': row['transaction_date'],
                    'account': row['account_name'],
                    'source': 'journal_entries.csv'
                })
            
            # Load orders for context
            orders_df = pd.read_csv('data/orders.csv')
            deliveries_df = pd.read_csv('data/deliveries.csv')
            
            # Merge orders with deliveries for cutoff analysis
            merged_df = orders_df.merge(deliveries_df, on='order_id', how='left')
            
            for _, row in merged_df.iterrows():
                content = f"Order {row['order_id']}: Customer {row['customer_id']} " \
                         f"ordered {row['quantity']} units on {row['order_date']}. " \
                         f"Total amount: ${row['total_amount']:.2f}. "
                
                if pd.notna(row['delivery_date']):
                    content += f"Delivered on {row['delivery_date']} " \
                              f"(Status: {row['delivery_status']}). "
                    
                    # Add cutoff analysis
                    order_year = pd.to_datetime(row['order_date']).year
                    delivery_year = pd.to_datetime(row['delivery_date']).year if pd.notna(row['delivery_date']) else None
                    
                    if delivery_year and order_year != delivery_year:
                        content += f"CUTOFF CONSIDERATION: Order placed in {order_year}, " \
                                  f"delivered in {delivery_year}. "
                else:
                    content += "Not yet delivered. "
                
                documents.append({
                    'content': content,
                    'type': 'order',
                    'order_id': row['order_id'],
                    'order_date': row['order_date'],
                    'delivery_date': row.get('delivery_date'),
                    'source': 'orders_deliveries.csv'
                })
                
        except Exception as e:
            print(f"Error loading transaction data: {e}")
        
        return documents
    
    def build_index(self):
        """Build the FAISS index from all documents"""
        print("Building RAG index...")
        
        # Load all documents
        policy_docs = self.load_policy_text()
        transaction_docs = self.load_transaction_examples()
        
        all_docs = policy_docs + transaction_docs
        print(f"Total documents: {len(all_docs)}")
        
        if not all_docs:
            print("No documents found to index")
            return
        
        # Extract text content and metadata
        texts = [doc['content'] for doc in all_docs]
        self.metadata = all_docs
        
        # Generate embeddings
        print("Generating embeddings...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add embeddings to index
        self.index.add(embeddings.astype('float32'))
        
        print(f"Index built with {self.index.ntotal} documents")
        print(f"Embedding dimension: {dimension}")
    
    def save_index(self, index_dir: str = "rag/index"):
        """Save the FAISS index and metadata"""
        os.makedirs(index_dir, exist_ok=True)
        
        # Save FAISS index
        index_path = os.path.join(index_dir, "faiss_index.idx")
        faiss.write_index(self.index, index_path)
        
        # Save metadata
        metadata_path = os.path.join(index_dir, "metadata.pkl")
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)
        
        # Save model name for consistency
        model_path = os.path.join(index_dir, "model_name.txt")
        with open(model_path, 'w') as f:
            f.write(self.model._modules['0'].auto_model.name_or_path)
        
        print(f"Index saved to {index_dir}")
    
    def test_search(self, query: str, k: int = 3):
        """Test the search functionality"""
        if self.index is None:
            print("Index not built yet")
            return
        
        print(f"\nTesting search for: '{query}'")
        
        # Encode query
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding.astype('float32'), k)
        
        print(f"Top {k} results:")
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.metadata):
                doc = self.metadata[idx]
                print(f"\n{i+1}. Score: {score:.4f}")
                print(f"Type: {doc['type']}")
                print(f"Source: {doc['source']}")
                print(f"Content: {doc['content'][:200]}...")

def main():
    """Main function to build the RAG index"""
    print("Starting RAG index building process...")
    
    # Initialize builder
    builder = RAGIndexBuilder()
    
    # Build index
    builder.build_index()
    
    # Save index
    builder.save_index()
    
    # Test searches
    test_queries = [
        "revenue cutoff policy",
        "when should revenue be recognized",
        "delivery before revenue",
        "period end cutoff violations",
        "journal entry for revenue"
    ]
    
    for query in test_queries:
        builder.test_search(query, k=2)
    
    print("\nRAG index building completed!")

if __name__ == "__main__":
    main()