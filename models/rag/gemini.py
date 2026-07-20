import os 
from google import genai
from dotenv import load_dotenv
load_dotenv()
api_key=os.getenv("GEMINI_API_KEY")
print('API Key Loaded:',api_key[:10]+'...')
client=genai.Client(api_key=api_key)
def generate_observation(caption, retrieved_documents):
    prompt = f"""
You are an expert astronomer.

Image Caption:
{caption}

Retrieved Astronomy Knowledge:
{retrieved_documents}

Using ONLY the retrieved information:

## Scientific Observation
Describe the astronomical object naturally in 2–3 paragraphs.

## Interesting Facts
Provide 3 concise bullet points about the object.

If the caption appears uncertain, explicitly mention the uncertainty instead of making assumptions.

Do not repeat the same facts multiple times.
Keep the total response under 250 words.
"""
    response = client.models.generate_content(model='gemini-flash-latest',contents=prompt)
    if isinstance(retrieved_documents, list):
        retrieved_documents = "\n".join(f"- {doc}" for doc in retrieved_documents)
    return response.text 

    
    

if __name__ == "__main__":
    caption = "The Orion Nebula"
    docs = [
        "The Orion Nebula is a stellar nursery approximately 1344 light years away.",
        "It is one of the brightest diffuse nebulae visible from Earth.",
        "It contains young stars and ionized hydrogen gas."
    ]
    output=generate_observation(caption,docs)
    print(output)
