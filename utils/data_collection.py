import os
import requests
from bs4 import BeautifulSoup
import PyPDF2
from sentence_transformers import SentenceTransformer

# Initialize embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def download_pdfs(url_list, save_dir="data/pdfs"):
    """
    Downloads PDFs from a list of URLs and saves them to the specified directory.
    """
    os.makedirs(save_dir, exist_ok=True)
    for url in url_list:
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise an exception for bad responses (4xx, 5xx)
            filename = os.path.join(save_dir, url.split('/')[-1])
            with open(filename, 'wb') as f:
                f.write(response.content)
            print(f"Downloaded: {filename}")
        except Exception as e:
            print(f"Failed to download {url}: {e}")

def extract_text_from_pdfs(pdf_dir):
    """
    Extracts text from PDFs in the given directory and returns a list of documents.
    Each document is a dictionary with 'text', 'source', and 'embeddings'.
    """
    documents = []
    for filename in os.listdir(pdf_dir):
        if filename.endswith('.pdf'):
            path = os.path.join(pdf_dir, filename)
            try:
                with open(path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    text = ""
                    for page in reader.pages:
                        page_text = page.extract_text()
                        if page_text:  # Ensure there is text on the page
                            text += page_text
                    if text.strip():  # Only add documents with content
                        document = {
                            'text': text,
                            'source': filename,
                            'embeddings': embedding_model.encode(text)  # Generate embeddings
                        }
                        documents.append(document)
                    else:
                        print(f"Skipped empty PDF: {filename}")
            except Exception as e:
                print(f"Failed to extract text from {filename}: {e}")
    return documents
