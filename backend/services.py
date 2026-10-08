import re
from models.pipeline import analyze_astronomy_image

SUMMARY_CHARS = 320


def _split_observation(text):
    """Split Gemini's markdown into (observation paragraphs, list of fact bullets)."""
    text = text or ""
    obs_match = re.search(
        r"##\s*Scientific Observation\s*(.*?)(?=##\s*Interesting Facts|\Z)",
        text, re.S | re.I)
    facts_match = re.search(r"##\s*Interesting Facts\s*(.*)", text, re.S | re.I)
    observation = obs_match.group(1).strip() if obs_match else text.strip()
    facts = []
    if facts_match:
        for line in facts_match.group(1).splitlines():
            line = re.sub(r"^\s*[\*\-•]\s+", "", line).strip()
            if line:
                facts.append(line)
    return observation, facts


def _to_card(item):
    """Turn a retrieved 'Title:..\nExplanation:\n..' document into a UI card."""
    text = item.get("text", "")
    match = re.match(r"Title:(.*?)\nExplanation:\n(.*)", text, re.S)
    title = item.get("title") or (match.group(1).strip() if match else "Astronomy knowledge")
    body = match.group(2).strip() if match else text.strip()
    if len(body) > SUMMARY_CHARS:
        body = body[:SUMMARY_CHARS].rsplit(" ", 1)[0] + "…"
    return {
        "title": title,
        "summary": body,
        "date": item.get("date", ""),
        "source_url": item.get("source_url", ""),
    }


def analyze_image(image_path):
    result = analyze_astronomy_image(image_path)
    observation, facts = _split_observation(result["observation"])
    return {
	"recognition":result.get("recognition",[]),
        "caption": result["caption"],
        "observation": result["observation"],
        "scientific_observation": observation,
        "facts": facts,
        "documents": result["documents"],
        "knowledge": [_to_card(k) for k in result.get("knowledge", [])],
    }
