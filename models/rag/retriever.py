from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb
PROJECT_ROOT=Path(__file__).resolve().parent.parent.parent
CHROMA_DIR=PROJECT_ROOT/'data'/'processed'/'chroma'
def load_retriever():
    print('Loading Sentence-BERT model')
    embedding_model=SentenceTransformer('all-MiniLM-L6-v2')
    print('Loading ChromaDB')
    client=chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection=client.get_collection('astronomy_knowledge')
    return embedding_model,collection
embedding_model,collection=load_retriever()
def retrieve_documents(query,top_k=5):
    print(f'Searching for: {query}\n')
    query_embedding = embedding_model.encode(query).tolist()
    results=collection.query(query_embeddings=[query_embedding],n_results=top_k)
    return results['documents'][0]
def display_results(results):
    documents=results['documents'][0]
    print('    TOP MATCHES    ')
    for i, doc in enumerate(documents,start=1):
        print(f'\n Result {i}\n')
        print(doc)
if __name__=='__main__':
    while True:
        query=input("\n Enter an astronomy caption(or 'exit'):")
        if query.lower()=='exit':
            break
        results=retrieve_documents(
            query,top_k=5)
        display_results(results)