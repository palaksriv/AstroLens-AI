import re
from pathlib import Path
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "rag"
NASA_CSV = RAW_DATA_DIR / "nasa_apod_complete.csv"
CHROMA_DIR = PROJECT_ROOT / "data" / "processed" / "chroma"
COLLECTION_NAME = 'astronomy_knowledge'
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'

# Everything from these APOD footer markers onwards is site boilerplate
# ("We keep an archive of ... brought to you by Robert Nemiroff ...").
FOOTER_MARKERS = [
    "We keep an archive",
    "Astronomy Picture of the Day is brought to you by",
    "Original material on this page is copyrighted",
]
HEADER_PHRASES = ["Today's Picture:", "Explanation:"]


def clean_text(text):
    if pd.isna(text):
        return ''
    # Normalise whitespace first: the raw data contains double spaces
    # ("We keep an  archive file."), which broke plain str.replace matching.
    text = ' '.join(str(text).split())
    for phrase in HEADER_PHRASES:
        text = text.replace(phrase, '')
    for marker in FOOTER_MARKERS:
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx]
    return ' '.join(text.split())


def apod_url(date):
    """APOD pages are named apYYMMDD.html, e.g. 1995-06-16 -> ap950616.html"""
    if pd.isna(date):
        return ''
    match = re.match(r'(\d{4})-(\d{2})-(\d{2})', str(date))
    if not match:
        return ''
    year, month, day = match.groups()
    return f"https://apod.nasa.gov/apod/ap{year[2:]}{month}{day}.html"


def build_vector_database():
    print('Loading Sentence-BERT model')
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    print('Reading NASA APOD dataset')
    df = pd.read_csv(NASA_CSV)
    print(f'Loaded {len(df)} astronomy documents')
    documents, metadatas, ids = [], [], []
    for idx, row in df.iterrows():
        explanation = clean_text(row['explanation'])
        if not explanation:  # a few rows have no explanation at all
            continue
        title = str(row['title']).strip()
        documents.append(f"Title:{title}\nExplanation:\n{explanation}")
        metadatas.append({
            'title': title,
            'date': '' if pd.isna(row['date']) else str(row['date']),
            'source_url': apod_url(row['date']),
            'media_type': '' if pd.isna(row['media_type']) else str(row['media_type']),
        })
        ids.append(str(idx))
    print(f'Skipped {len(df) - len(documents)} rows with empty explanations')
    print('\n Sample Document:\n')
    print(documents[0])
    print(f'Total documents: {len(documents)}')
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    try:
        client.delete_collection(COLLECTION_NAME)
        print('\n Old Collection deleted')

    except Exception:
        print('\n No previous collection found')
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    print('\n Generating embeddings')
    embeddings = embedding_model.encode(documents, show_progress_bar=True)
    print("Embeddings generated successfully")
    print('Saving embeddings to ChromaDB')
    BATCH_SIZE = 5000

    for start in range(0, len(documents), BATCH_SIZE):
        end = start + BATCH_SIZE

        collection.add(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
            embeddings=embeddings[start:end].tolist()
        )

        print(f"Stored documents {start} to {min(end, len(documents))}")
    print(f'\n Stored {collection.count()} documents in ChromaDB.')
    print(f'\n Vector database built successfully')


if __name__ == '__main__':
    build_vector_database()
