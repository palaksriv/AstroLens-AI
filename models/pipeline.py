from models.captioning.inference import generate_caption
from models.rag.retriever import retrieve_documents
from models.rag.gemini import generate_observation
def analyze_astronomy_image(image_path):
    print('Generating caption...')
    caption=generate_caption(image_path)
    print('Caption:')
    print(caption)
    print('\nRetrieving relevant documents...')
    docs=retrieve_documents(caption)
    print(f'Retrieved {len(docs)} documents')
    print('\nGenerating scientific observation...')
    observation=generate_observation(caption,docs)
    return {
        'caption':caption,
        'documents':docs,
        'observation':observation
    }
if __name__=='__main__':
    image_path='sample_images/images (1).jpeg'
    result=analyze_astronomy_image(image_path)
    print('\n Caption:')
    print(result['caption'])
    print('\n Observation:')
    print(result['observation'])