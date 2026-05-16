"""
generate_data.py — Generate synthetic gate training data via LLM.

Uses Ollama (testing) or Groq (production) to generate 150 example queries
per expert class (text, table, image) = 450 total labeled samples.
Output: data/gate_training.json
"""

import json
import os
import sys
import requests

# Add parent dirs to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from engine.config import (
    TESTING, OLLAMA_BASE_URL, OLLAMA_MODEL,
    GROQ_API_KEY, GROQ_MODEL, GATE_TRAINING_DATA_PATH
)


PROMPT = """You are a data generator for a query routing classifier.
I need training data to classify user queries into four retrieval expert categories:
- TEXT: queries about prose, clauses, descriptions, explanations, summaries
- TABLE: queries about numbers, comparisons, statistics, rows, columns, data
- IMAGE: queries about diagrams, charts, figures, photos, visual content
- CODE: queries about source code, logic, variables, functions, classes, and programming implementation

Here are seed examples:

TEXT queries:
- "summarize the indemnity clause"
- "explain how the approval process works"
- "what does the contract say about termination"
- "describe the methodology section"
- "what are the key findings in the conclusion"

TABLE queries:
- "how many rows have revenue > 100k"
- "what was the average Q3 sales"
- "compare expenses between 2022 and 2023"
- "show me the top 5 products by unit sales"
- "what percentage of budget went to marketing"

IMAGE queries:
- "describe the architecture diagram on page 3"
- "what does the flowchart show about the process"
- "explain the bar chart comparing quarterly results"
- "what information is in the pie chart"
- "describe the system diagram"

CODE queries:
- "where is the authentication middleware defined?"
- "how does the database connection retry logic work?"
- "find the function that handles user login"
- "show me the definition of the QueryRequest class"
- "what are the parameters for the queryStream function"

Generate exactly 50 diverse queries for EACH category (200 total).
Make them varied — different domains (legal, finance, medical, engineering, academic).
Make some ambiguous (could route to multiple experts).

Return ONLY valid JSON in this exact format, no other text:
{"text": ["query1", "query2", ...], "table": ["query1", "query2", ...], "image": ["query1", "query2", ...], "code": ["query1", "query2", ...]}
"""


def generate_via_ollama() -> dict:
    """Generate training data using Ollama."""
    print(f"[Gate] Generating synthetic data via Ollama ({OLLAMA_MODEL})...")
    
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": OLLAMA_MODEL,
            "prompt": PROMPT,
            "stream": False,
            "options": {
                "temperature": 0.8,
                "num_predict": 16384,
            }
        },
        timeout=600  # 10 min timeout — generating 450 queries takes a while
    )
    response.raise_for_status()
    
    raw = response.json()["response"]
    
    # Extract JSON from response (handle markdown code blocks)
    if "```json" in raw:
        raw = raw.split("```json")[1].split("```")[0]
    elif "```" in raw:
        raw = raw.split("```")[1].split("```")[0]
    
    data = json.loads(raw.strip())
    return data


def generate_via_groq() -> dict:
    """Generate training data using Groq API."""
    print(f"[Gate] Generating synthetic data via Groq ({GROQ_MODEL})...")
    
    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": GROQ_MODEL,
            "messages": [{"role": "user", "content": PROMPT}],
            "temperature": 0.8,
            "max_tokens": 8000,
        },
        timeout=120
    )
    response.raise_for_status()
    
    raw = response.json()["choices"][0]["message"]["content"]
    
    if "```json" in raw:
        raw = raw.split("```json")[1].split("```")[0]
    elif "```" in raw:
        raw = raw.split("```")[1].split("```")[0]
    
    data = json.loads(raw.strip())
    return data


def generate_via_gemini() -> dict:
    """Generate training data using Gemini API."""
    print(f"[Gate] Generating synthetic data via Gemini...")
    from engine.config import GEMINI_API_KEY, GEMINI_BASE_URL
    
    response = requests.post(
        f"{GEMINI_BASE_URL}/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}",
        headers={"Content-Type": "application/json"},
        json={
            "contents": [{"parts": [{"text": PROMPT}]}],
            "generationConfig": {
                "temperature": 0.8,
                "maxOutputTokens": 8000,
                "responseMimeType": "application/json"
            }
        },
        timeout=120
    )
    response.raise_for_status()
    
    raw = response.json()["candidates"][0]["content"]["parts"][0]["text"]
    
    if "```json" in raw:
        raw = raw.split("```json")[1].split("```")[0]
    elif "```" in raw:
        raw = raw.split("```")[1].split("```")[0]
    
    data = json.loads(raw.strip())
    return data


def validate_data(data: dict) -> bool:
    """Validate the generated data structure."""
    required_keys = {"text", "table", "image", "code"}
    if not required_keys.issubset(data.keys()):
        print(f"[Gate] ERROR: Missing keys. Got {data.keys()}, need {required_keys}")
        return False
    
    for key in required_keys:
        if not isinstance(data[key], list):
            print(f"[Gate] ERROR: '{key}' is not a list")
            return False
        if len(data[key]) < 10:
            print(f"[Gate] WARNING: '{key}' only has {len(data[key])} queries (expected ~150)")
    
    return True


def main():
    if os.path.exists(GATE_TRAINING_DATA_PATH):
        print(f"[Gate] Training data already exists at {GATE_TRAINING_DATA_PATH}")
        overwrite = input("Overwrite? (y/N): ").strip().lower()
        if overwrite != "y":
            print("[Gate] Skipping generation.")
            return
    
    # Generate data
    if TESTING:
        data = generate_via_ollama()
    else:
        data = generate_via_gemini()
    
    # Validate
    if not validate_data(data):
        print("[Gate] Generated data failed validation. Saving anyway for inspection.")
    
    # Report stats
    for key in ["text", "table", "image", "code"]:
        count = len(data.get(key, []))
        print(f"[Gate]   {key}: {count} queries")
    
    # Save
    os.makedirs(os.path.dirname(GATE_TRAINING_DATA_PATH), exist_ok=True)
    with open(GATE_TRAINING_DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"[Gate] Saved to {GATE_TRAINING_DATA_PATH}")
    print(f"[Gate] Total: {sum(len(v) for v in data.values())} queries")


if __name__ == "__main__":
    main()
