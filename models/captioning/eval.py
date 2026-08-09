import csv
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from transformers import BlipProcessor, BlipForConditionalGeneration
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from evaluate import load
from tqdm import tqdm
from dataset import AstroLensDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 4
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
smooth = SmoothingFunction().method1
rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
meteor = load("meteor")
test_dataset = AstralVisionDataset(csv_file="test.csv",evaluation=True)

def collate_fn(batch):
    images, captions, image_names = zip(*batch)
    return list(images), list(captions), list(image_names)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn,
)
def load_model(model_type):
    if model_type == "pretrained":
        processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
    else:

        model_path = PROJECT_ROOT / "models" / "fine_tuned_blip"

        processor = BlipProcessor.from_pretrained(model_path)

        model = BlipForConditionalGeneration.from_pretrained(model_path)

    model.to(DEVICE)
    model.eval()
    return processor, model

def evaluate(model_type):
    processor, model = load_model(model_type)
    predictions = []
    references = []
    filenames = []
    for images, captions, image_names in tqdm(test_loader, desc=model_type):

        inputs = processor(
            images=images,
            return_tensors="pt",
            padding=True,
        )

        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
            )

        decoded = processor.batch_decode(
            outputs,
            skip_special_tokens=True,
        )

        predictions.extend(decoded)
        references.extend(captions)
        filenames.extend(image_names)
    bleu_scores = []
    rouge_scores = []

    for pred, ref in zip(predictions, references):

        bleu_scores.append(
            sentence_bleu(
                [ref.split()],
                pred.split(),
                smoothing_function=smooth,
            )
        )

        rouge_scores.append(
            rouge.score(ref, pred)["rougeL"].fmeasure
        )

    meteor_score = meteor.compute(
        predictions=predictions,
        references=references,
    )["meteor"]

    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    avg_rouge = sum(rouge_scores) / len(rouge_scores)

    csv_path = RESULTS_DIR / f"{model_type}_predictions.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:

        writer = csv.writer(f)

        writer.writerow(
            ["Image", "Ground Truth", "Prediction"]
        )

        for name, gt, pred in zip(
            filenames,
            references,
            predictions,
        ):
            writer.writerow([name, gt, pred])

    return avg_bleu, avg_rouge, meteor_score

pre_bleu, pre_rouge, pre_meteor = evaluate("pretrained")

fine_bleu, fine_rouge, fine_meteor = evaluate("finetuned")

with open(
    RESULTS_DIR / "comparison.csv",
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.writer(f)

    writer.writerow(
        ["Model", "BLEU", "ROUGE-L", "METEOR"]
    )

    writer.writerow(
        ["Pretrained", pre_bleu, pre_rouge, pre_meteor]
    )

    writer.writerow(
        ["Fine-tuned", fine_bleu, fine_rouge, fine_meteor]
    )

print('Evaluation Results:\n')
print(f"Pretrained  BLEU: {pre_bleu:.4f}")
print(f"Pretrained  ROUGE-L: {pre_rouge:.4f}")
print(f"Pretrained  METEOR: {pre_meteor:.4f}\n")

print(f"Fine-tuned  BLEU: {fine_bleu:.4f}")
print(f"Fine-tuned  ROUGE-L: {fine_rouge:.4f}")
print(f"Fine-tuned  METEOR: {fine_meteor:.4f}")

print(f"\nResults saved to: {RESULTS_DIR}")