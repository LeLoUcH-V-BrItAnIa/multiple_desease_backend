from flask import Flask, request, jsonify
import google.generativeai as genai
from recommender import build_prompt
import re, json
import os
from dotenv import load_dotenv


# Load .env file
load_dotenv()





app = Flask(__name__)
# Configure Gemini API
# Set your Gemini API key
# genai.configure(api_key="")
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("models/gemma-3n-e2b-it")

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
    app.run(debug=True)