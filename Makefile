.PHONY: install setup run clean test

# Install dependencies
install:
	pip install -r requirements.txt

# Setup the complete environment
setup: install
	python csv_to_rdf.py
	python rag/build_index.py

# Run the Streamlit application
run:
	streamlit run app.py

# Clean generated files
clean:
	rm -rf data/*.ttl
	rm -rf rag/index/
	rm -rf __pycache__/
	find . -name "*.pyc" -delete

# Test the setup
test:
	python -c "import streamlit, pandas, rdflib, pyshacl, faiss, sentence_transformers; print('All dependencies OK')"

# Help
help:
	@echo "Available targets:"
	@echo "  install - Install Python dependencies"
	@echo "  setup   - Full setup including data transformation and index building"
	@echo "  run     - Run the Streamlit application"
	@echo "  clean   - Clean generated files"
	@echo "  test    - Test dependencies"
	@echo "  help    - Show this help"