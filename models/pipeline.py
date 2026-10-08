from models.captioning.inference import generate_caption
from models.recognition.clip import recognize_object
from models.rag.retriever import retrieve_knowledge
from models.rag.gemini import generate_observation


def analyze_astronomy_image(image_path):
    print("Recognizing object with CLIP...")
    recognition = recognize_object(image_path)

    top_object = recognition[0]["label"]
    print(f"CLIP result: {top_object}")

    print("Generating caption with BLIP...")
    caption = generate_caption(image_path)

    print("Retrieving relevant documents...")
    knowledge = retrieve_knowledge(top_object)
    docs = [item["text"] for item in knowledge]

    print(f"Retrieved {len(docs)} documents")

    print("Generating scientific observation...")
    observation = generate_observation(
        f"{top_object}. BLIP caption: {caption}",
        docs
    )

    return {
        "recognition": recognition,
        "caption": caption,
        "documents": docs,
        "knowledge": knowledge,
        "observation": observation,
    }


if __name__ == "__main__":
    image_path = "sample_images/images (1).jpeg"
    result = analyze_astronomy_image(image_path)

    print("\nRecognition:")
    print(result["recognition"])

    print("\nCaption:")
    print(result["caption"])

    print("\nObservation:")
    print(result["observation"])