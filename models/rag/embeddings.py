from pathlib import Path
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
PROJECT_ROOT=Path(__file__).resolve().parent.parent.parent
RAW_DATA_DIR=PROJECT_ROOT/"data"/"raw"/"rag"
NASA_CSV=RAW_DATA_DIR/"nasa_apod_complete.csv"
CHROMA_DIR=PROJECT_ROOT/"data"/"processed"/"chroma"
def clean_text(text):
    if pd.isna(text):
        return ''
    text=str(text)
    remove_phrases=["Today's Picture:",
                    "Explanation:",
                    "Astronomy Picture of the Day",
                    "We keep an archive",
                    "Original material on this page is copyrighted",
                    "We keep an archive",
                    "We keep an archive file.",
                    "Original material on this page is copyrighted",
                    "is brought to you by",
                    "Robert Nemiroff",
                    "Jerry Bonnell"]
    for phrase in remove_phrases:
        text=text.replace(phrase,'')
    return ' '.join(text.split())
def build_vector_database():
    print('Loading Sentence-BERT model')
    embedding_model=SentenceTransformer('all-MiniLM-L6-v2')
    print('Reading NASA APOD dataset')
    df=pd.read_csv(NASA_CSV)
    print(f'Loaded {len(df)} astronomy documents')
    documents=[]
    for _, row in df.iterrows():
        title=row['title']
        explanation=clean_text(row['explanation'])
        document=f"""Title:{title}
Explanation:
{explanation}"""
        documents.append(document)
    print('\n Sample Document:\n')
    print(documents[0])
    print(f'Total documents: {len(documents)}')
    CHROMA_DIR.mkdir(parents=True,exist_ok=True)
    client=chromadb.PersistentClient(path=str(CHROMA_DIR))

    try:
        client.delete_collection('astronomy_knowledge')
        print('\n Old Collection deleted')

    except Exception:
        print('\n No previous collection found')
    collection=client.get_or_create_collection(name='astronomy_knowledge')
    print('\n Generating embeddings')
    embeddings=embedding_model.encode(documents,show_progress_bar=True)
    print("Embeddings generated successfully")
    ids=[str(i) for i in range(len(documents))]
    print('Saving embeddings to ChromaDB')
    BATCH_SIZE = 5000

    for start in range(0, len(documents), BATCH_SIZE):
        end = start + BATCH_SIZE

        collection.add(
            ids=ids[start:end],
            documents=documents[start:end],
            embeddings=embeddings[start:end].tolist()
        )

        print(f"Stored documents {start} to {min(end, len(documents))}")
    print(f'\n Stored {collection.count()} documents in ChromaDB.')
    print(f'\n Vector database built successfully')
if __name__=='__main__':
    build_vector_database()