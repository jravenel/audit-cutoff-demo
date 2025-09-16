#!/usr/bin/env python3
"""
ETL Script: Convert CSV data to RDF format
Transforms CSV files into RDF triples following the audit cutoff ontology
"""

import pandas as pd
from rdflib import Graph, Namespace, Literal, URIRef, BNode
from rdflib.namespace import RDF, RDFS, XSD
import os
from datetime import datetime

# Define namespaces
AC = Namespace("http://example.org/audit-cutoff#")
DATA = Namespace("http://example.org/audit-cutoff/data#")

def create_base_graph():
    """Create base RDF graph with ontology and namespaces"""
    g = Graph()
    
    # Load the ontology
    ontology_path = "ont/cutoff_ontology.ttl"
    if os.path.exists(ontology_path):
        g.parse(ontology_path, format="turtle")
    
    # Bind namespaces
    g.bind("", AC)
    g.bind("data", DATA)
    g.bind("rdfs", RDFS)
    
    return g

def add_accounts(g):
    """Add account entities to the graph"""
    # Revenue Account
    revenue_account = DATA.RevenueAccount
    g.add((revenue_account, RDF.type, AC.Account))
    g.add((revenue_account, AC.accountCode, Literal("4000")))
    g.add((revenue_account, AC.accountName, Literal("Revenue")))
    
    # Accounts Receivable Account  
    ar_account = DATA.AccountsReceivableAccount
    g.add((ar_account, RDF.type, AC.Account))
    g.add((ar_account, AC.accountCode, Literal("1200")))
    g.add((ar_account, AC.accountName, Literal("Accounts Receivable")))
    
    # Create a mapping for easy lookup
    AC.RevenueAccount = revenue_account
    AC.ARAccount = ar_account

def process_orders(g, orders_df):
    """Convert orders CSV to RDF"""
    print("Processing orders...")
    
    for _, row in orders_df.iterrows():
        order_uri = DATA[f"order_{row['order_id']}"]
        customer_uri = DATA[f"customer_{row['customer_id']}"]
        product_uri = DATA[f"product_{row['product_id']}"]
        
        # Order entity
        g.add((order_uri, RDF.type, AC.Order))
        g.add((order_uri, AC.orderId, Literal(row['order_id'])))
        g.add((order_uri, AC.orderDate, Literal(row['order_date'], datatype=XSD.date)))
        g.add((order_uri, AC.quantity, Literal(row['quantity'], datatype=XSD.integer)))
        g.add((order_uri, AC.unitPrice, Literal(row['unit_price'], datatype=XSD.decimal)))
        g.add((order_uri, AC.amount, Literal(row['total_amount'], datatype=XSD.decimal)))
        g.add((order_uri, AC.orderedBy, customer_uri))
        g.add((order_uri, AC.forProduct, product_uri))
        
        # Customer entity
        g.add((customer_uri, RDF.type, AC.Customer))
        
        # Product entity  
        g.add((product_uri, RDF.type, AC.Product))

def process_deliveries(g, deliveries_df):
    """Convert deliveries CSV to RDF"""
    print("Processing deliveries...")
    
    for _, row in deliveries_df.iterrows():
        delivery_uri = DATA[f"delivery_{row['delivery_id']}"]
        order_uri = DATA[f"order_{row['order_id']}"]
        
        # Delivery entity
        g.add((delivery_uri, RDF.type, AC.Delivery))
        g.add((delivery_uri, AC.deliveryDate, Literal(row['delivery_date'], datatype=XSD.date)))
        g.add((delivery_uri, AC.deliveryStatus, Literal(row['delivery_status'])))
        g.add((delivery_uri, AC.quantity, Literal(row['shipped_quantity'], datatype=XSD.integer)))
        g.add((delivery_uri, AC.description, Literal(row['delivery_address'])))
        
        # Link to order
        g.add((order_uri, AC.hasDelivery, delivery_uri))

def process_invoices(g, invoices_df):
    """Convert invoices CSV to RDF"""
    print("Processing invoices...")
    
    for _, row in invoices_df.iterrows():
        invoice_uri = DATA[f"invoice_{row['invoice_id']}"]
        order_uri = DATA[f"order_{row['order_id']}"]
        
        # Invoice entity
        g.add((invoice_uri, RDF.type, AC.Invoice))
        g.add((invoice_uri, AC.invoiceDate, Literal(row['invoice_date'], datatype=XSD.date)))
        g.add((invoice_uri, AC.amount, Literal(row['invoice_amount'], datatype=XSD.decimal)))
        g.add((invoice_uri, AC.description, Literal(f"Invoice {row['invoice_id']} for {row['order_id']}")))
        
        # Link to order
        g.add((order_uri, AC.hasInvoice, invoice_uri))

def process_journal_entries(g, je_df):
    """Convert journal entries CSV to RDF"""
    print("Processing journal entries...")
    
    for _, row in je_df.iterrows():
        je_uri = DATA[f"je_{row['journal_entry_id']}"]
        
        # Journal Entry entity
        g.add((je_uri, RDF.type, AC.JournalEntry))
        g.add((je_uri, AC.transactionDate, Literal(row['transaction_date'], datatype=XSD.date)))
        g.add((je_uri, AC.debitAmount, Literal(row['debit_amount'], datatype=XSD.decimal)))
        g.add((je_uri, AC.creditAmount, Literal(row['credit_amount'], datatype=XSD.decimal)))
        g.add((je_uri, AC.description, Literal(row['description'])))
        
        # Link to account based on account code
        if row['account_code'] == '4000':  # Revenue
            g.add((je_uri, AC.toAccount, DATA.RevenueAccount))
        elif row['account_code'] == '1200':  # Accounts Receivable
            g.add((je_uri, AC.toAccount, DATA.AccountsReceivableAccount))
        
        # Link to invoice if reference exists
        if pd.notna(row['reference']) and row['reference'].startswith('INV'):
            invoice_uri = DATA[f"invoice_{row['reference']}"]
            g.add((invoice_uri, AC.hasJournalEntry, je_uri))

def main():
    """Main ETL process"""
    print("Starting CSV to RDF conversion...")
    
    # Create base graph
    g = create_base_graph()
    
    # Add account entities
    add_accounts(g)
    
    # Load CSV files
    try:
        orders_df = pd.read_csv('data/orders.csv')
        deliveries_df = pd.read_csv('data/deliveries.csv')
        invoices_df = pd.read_csv('data/invoices.csv')
        je_df = pd.read_csv('data/journal_entries.csv')
    except FileNotFoundError as e:
        print(f"Error loading CSV file: {e}")
        print("Please ensure all CSV files exist in the data/ directory")
        return
    
    # Process each CSV file
    process_orders(g, orders_df)
    process_deliveries(g, deliveries_df)
    process_invoices(g, invoices_df)
    process_journal_entries(g, je_df)
    
    # Save the RDF graph
    output_file = "data/audit_data.ttl"
    g.serialize(destination=output_file, format="turtle")
    print(f"RDF data saved to {output_file}")
    print(f"Total triples: {len(g)}")

if __name__ == "__main__":
    main()