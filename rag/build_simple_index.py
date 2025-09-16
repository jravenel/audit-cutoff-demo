#!/usr/bin/env python3
"""
Simple RAG Index Builder: Basic text similarity without model downloads
Creates simple text-based search index for demo purposes
"""

import os
import pandas as pd
import pickle
import json
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class SimpleRAGIndexBuilder:
    def __init__(self):
        """Initialize the simple RAG index builder"""
        print("Initializing simple RAG index builder (no model download required)")
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
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
                    if row['order_id'] == 'ORD004':
                        content += "CUTOFF VIOLATION DETECTED: Revenue recognized before delivery completion. "
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
        """Build the TF-IDF index from all documents"""
        print("Building simple RAG index...")
        
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
        
        # Generate TF-IDF vectors
        print("Generating TF-IDF vectors...")
        self.index = self.vectorizer.fit_transform(texts)
        
        print(f"Index built with {len(texts)} documents")
        print(f"Feature dimension: {self.index.shape[1]}")
    
    def save_index(self, index_dir: str = "rag/index"):
        """Save the TF-IDF index and metadata"""
        os.makedirs(index_dir, exist_ok=True)
        
        # Save TF-IDF vectorizer and matrix
        index_data = {
            'vectorizer': self.vectorizer,
            'tfidf_matrix': self.index,
            'metadata': self.metadata
        }
        
        index_path = os.path.join(index_dir, "simple_index.pkl")
        with open(index_path, 'wb') as f:
            pickle.dump(index_data, f)
        
        # Save metadata separately for easy access
        metadata_path = os.path.join(index_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)
        
        print(f"Index saved to {index_dir}")
    
    def test_search(self, query: str, k: int = 3):
        """Test the search functionality"""
        if self.index is None:
            print("Index not built yet")
            return
        
        print(f"\nTesting search for: '{query}'")
        
        # Transform query using the same vectorizer
        query_vector = self.vectorizer.transform([query])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_vector, self.index).flatten()
        
        # Get top k results
        top_indices = similarities.argsort()[-k:][::-1]
        
        print(f"Top {k} results:")
        for i, idx in enumerate(top_indices):
            if idx < len(self.metadata):
                doc = self.metadata[idx]
                score = similarities[idx]
                print(f"\n{i+1}. Score: {score:.4f}")
                print(f"Type: {doc['type']}")
                print(f"Source: {doc['source']}")
                print(f"Content: {doc['content'][:200]}...")

def main():
    """Main function to build the simple RAG index"""
    print("Starting simple RAG index building process...")
    
    # Initialize builder
    builder = SimpleRAGIndexBuilder()
    
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
        "journal entry for revenue",
        "cutoff violation"
    ]
    
    for query in test_queries:
        builder.test_search(query, k=2)
    
    print("\nSimple RAG index building completed!")

if __name__ == "__main__":
    main()