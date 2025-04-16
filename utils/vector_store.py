import chromadb
from chromadb.utils import embedding_functions

# Initialize Sentence Transformer embedding function
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Initialize persistent Chroma client and collection
client = chromadb.PersistentClient(path="agribot_db")
collection = client.get_or_create_collection(
    name="agriculture_resources",
    embedding_function=sentence_transformer_ef
)

def add_documents(documents):
    """
    Add documents to the Chroma database.
    Each document is expected to be a dictionary with 'text' and 'source' keys.
    """
    ids = [str(i) for i in range(len(documents))]
    collection.add(
        documents=[doc['text'] for doc in documents],
        metadatas=[{"source": doc['source']} for doc in documents],
        ids=ids
    )

def query_documents(query_texts, n_results=3):
    """
    Query the Chroma database for documents relevant to the query_texts.
    Returns the top n_results documents.
    """
    # Query the collection for relevant documents
    results = collection.query(
        query_texts=query_texts,
        n_results=n_results
    )
    return results
