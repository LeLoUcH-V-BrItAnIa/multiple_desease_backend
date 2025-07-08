def build_prompt(symptoms, mood, age, gender):
    return f"""
You are a professional health assistant AI.

Patient profile:
- Age: {age}
- Gender: {gender}
- Mood: {mood}
- Symptoms: {', '.join(symptoms)}

Your job is to give 3 helpful health suggestions in structured JSON only.

Respond in **ONLY this format** (no markdown, no explanations, just pure JSON):

{{
  "diet_tips": ["..."],
  "lifestyle_tips": ["..."],
  "notes": ["..."]
}}
"""