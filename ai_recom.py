import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from flask import Flask, request, jsonify
import google.generativeai as genai
from recommender import build_prompt
import re, json
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np



# Load .env file
load_dotenv()
rag_model = SentenceTransformer('all-MiniLM-L6-v2')
rag_index = faiss.read_index("./RAG/medical_index.faiss")
rag_docs = np.load("./RAG/medical_docs.npy", allow_pickle=True)

app = Flask(__name__)
# Configure Gemini API
# Set your Gemini API key
# genai.configure(api_key="")
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("models/gemma-3n-e2b-it")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    query = data.get("query")

    # RAG search
    query_vec = rag_model.encode([query])
    D, I = rag_index.search(query_vec, k=3)

    context = "\n".join([rag_docs[i] for i in I[0]])

    prompt = f"""
    You are a medical assistant AI.

    Rules:
    - Only answer from context
    - Do not guess
    - Do not give prescriptions
    - Always say: Consult a healthcare professional

    Context:
    {context}

    Question:
    {query}
    """

    response = model.generate_content(prompt)

    return jsonify({"response": response.text})

@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.json
    symptoms = data.get("symptoms", [])
    mood = data.get("mood", "")
    age = data.get("age", "")
    gender = data.get("gender", "")

    prompt = build_prompt(symptoms, mood, age, gender)

    try:
        response = model.generate_content(prompt)
        # Extract valid JSON from Gemini response
        json_response = extract_json(response.text)
        return jsonify(json_response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def extract_json(text):
    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        return json.loads(match.group()) if match else {
            "diet_tips": [],
            "lifestyle_tips": [],
            "notes": [],
            "error": "No valid JSON found",
            "raw_response": text
        }
    except Exception as e:
        return {
            "diet_tips": [],
            "lifestyle_tips": [],
            "notes": [],
            "error": f"JSON parse error: {str(e)}",
            "raw_response": text
        }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
    # app.run(debug=True, host="")