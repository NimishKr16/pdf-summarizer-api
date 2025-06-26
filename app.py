# app.py
# app.py
from dotenv import load_dotenv
load_dotenv()
from flask import Flask, request, jsonify
from flask_cors import CORS
import fitz  # PyMuPDF
import os
import requests
import time
import openai  # SDK for OpenRouter

from together import Together # SDK for Together AI

# Cooldown tracking for API providers
cooldowns = {
    "groq": 0,
    "openrouter": 0,
    "together": 0,
}

app = Flask(__name__)
# app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5 MB limit
CORS(app)  # Allow requests from all origins
client = Together()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
# TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

openai.api_key = OPENROUTER_API_KEY
openai.base_url = "https://openrouter.ai/api/v1"

# Initialize Together SDK
# together.api_key = TOGETHER_API_KEY

def extract_text_from_pdf(file_stream):
    doc = fitz.open(stream=file_stream.read(), filetype="pdf")
    return "\n".join([page.get_text() for page in doc])

def chunk_text(text, max_chars=3000):
    paragraphs = text.split("\n")
    chunks = []
    current = ""
    for p in paragraphs:
        if len(current) + len(p) < max_chars:
            current += p + "\n"
        else:
            chunks.append(current.strip())
            current = p + "\n"
    if current:
        chunks.append(current.strip())
    return chunks

def summarize_chunk(chunk):
    now = time.time()
    prompt = [
        {"role": "system", "content": "You are a helpful assistant. Summarize the following PDF content clearly and concisely, preserving important details."},
        {"role": "user", "content": f"PDF content:\n\n{chunk}\n\nPlease provide a summary."}
    ]

    # Try Groq
    if now > cooldowns["groq"]:
        try:
            headers = {
                "Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}",
                "Content-Type": "application/json"
            }
            payload = {"model": "llama-3.3-70b-versatile", "messages": prompt}
            res = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
            res.raise_for_status()
            print("Used Groq")
            return res.json()['choices'][0]['message']['content']
        except requests.exceptions.HTTPError as e:
            if res.status_code == 429:
                cooldowns["groq"] = now + 20 * 60
            print(f"Groq failed: {e}")

    # Try OpenRouter
    if now > cooldowns["openrouter"]:
        try:
            response = openai.ChatCompletion.create(
                model="meta-llama/llama-3.3-8b-instruct:free",
                messages=prompt
            )
            print("Used OpenRouter")
            return response['choices'][0]['message']['content']
        except openai.error.OpenAIError as e:
            if hasattr(e, 'http_status') and e.http_status == 429:
                cooldowns["openrouter"] = now + 20 * 60
            print(f"OpenRouter failed: {e}")

    # Try Together
    if now > cooldowns["together"]:
        try:
            response = client.chat.completions.create(
                model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
                messages=prompt
            )
            print("Used Together")
            return response.choices[0].message.content
        except Exception as e:
            if "429" in str(e):
                cooldowns["together"] = now + 20 * 60
            print(f"Together failed: {e}")

    raise RuntimeError("All providers failed or are in cooldown.")

@app.route("/summarize-pdf", methods=["POST"])
def summarize_pdf():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    file = request.files['file']

    # Check for page count limit
    file.stream.seek(0)
    doc = fitz.open(stream=file.read(), filetype="pdf")
    if len(doc) > 15:
        return jsonify({"error": "PDF exceeds 15 page limit."}), 400
    file.stream.seek(0)

    text = extract_text_from_pdf(file)
    chunks = chunk_text(text)
    print(f"Parsed {len(chunks)} chunks from the PDF.")

    summaries = []
    for i, chunk in enumerate(chunks):
        print(f"Summarizing chunk {i + 1}/{len(chunks)}...")
        try:
            summary = summarize_chunk(chunk)
            summaries.append(summary)
        except Exception as e:
            print(f"Error summarizing chunk {i + 1}: {e}")
            summaries.append("[Summary failed for this chunk]")

    final_summary = "\n\n".join(summaries)
    return jsonify({"summary": final_summary})

if __name__ == "__main__":
    app.run(debug=True, port=8000)