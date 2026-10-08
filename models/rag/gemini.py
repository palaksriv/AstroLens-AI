import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

GEMINI_MODEL = "gemini-3.8-flash"
_client = None


def _get_client():
    global _client

    if _client is None:
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set."
            )

        _client = genai.Client(api_key=api_key)

    return _client


def generate_observation(caption, retrieved_documents):

    if isinstance(retrieved_documents, (list, tuple)):
        retrieved_documents = "\n\n".join(
            f"- {doc}" for doc in retrieved_documents
        )

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

If the caption appears uncertain, explicitly mention the uncertainty.

Do not invent facts that are not supported by the caption or retrieved knowledge.

Keep the total response under 250 words.
"""

    try:
        interaction = _get_client().interactions.create(
            model=GEMINI_MODEL,
            input=prompt,
        )

        return interaction.output_text or ""

    except Exception as exc:
        print(f"Gemini unavailable: {exc}")

        return (
            "Gemini scientific analysis is temporarily unavailable. "
            "The recognition, caption and retrieved astronomy knowledge "
            "are still available below."
        )