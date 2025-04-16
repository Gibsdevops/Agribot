
from utils.data_collection import extract_text_from_pdfs
from utils.vector_store import add_documents

# Step 1: Extract text & embeddings from PDFs
documents = extract_text_from_pdfs("data/pdfs")

# Step 2: Add them to the vector database
add_documents(documents)

print("Data preparation complete! Your documents are now indexed.")
