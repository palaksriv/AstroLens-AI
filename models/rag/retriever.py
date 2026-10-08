from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CHROMA_DIR = PROJECT_ROOT / 'data' / 'processed' / 'chroma'
COLLECTION_NAME = 'astronomy_knowledge'
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'


def load_retriever():
    print('Loading Sentence-BERT model')
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    print('Loading ChromaDB')
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception as exc:
        raise RuntimeError(
            f"ChromaDB collection '{COLLECTION_NAME}' not found in {CHROMA_DIR}.\n"
            "Build it first with:  python -m models.rag.embeddings"
        ) from exc
    return embedding_model, collection


embedding_model, collection = load_retriever()


def retrieve_knowledge(query, top_k=5):
    """Return the top_k matches as dicts: text, title, date, source_url."""
    print(f'Searching for: {query}\n')
    query_embedding = embedding_model.encode(query).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    items = []
    for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
        meta = meta or {}  # collections built before metadata was added
        items.append({
            'text': doc,
            'title': meta.get('title', ''),
            'date': meta.get('date', ''),
            'source_url': meta.get('source_url', ''),
        })
    return items


def retrieve_documents(query, top_k=5):
    """Plain document texts only (kept for backwards compatibility)."""
    return [item['text'] for item in retrieve_knowledge(query, top_k)]


def display_results(documents):
    print('    TOP MATCHES    ')
    for i, doc in enumerate(documents, start=1):
        print(f'\n Result {i}\n')
        print(doc)


if __name__ == '__main__':
    while True:
        query = input("\n Enter an astronomy caption(or 'exit'):")
        if query.lower() == 'exit':
            break
        docs = retrieve_documents(query, top_k=5)
        display_results(docs)
