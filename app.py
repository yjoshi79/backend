from flask import Flask, request, jsonify
from flask_cors import CORS
from huggingface_hub import InferenceClient
import os

app = Flask(__name__)
CORS(app)

client = InferenceClient(
    model="mistralai/Mistral-7B-Instruct-v0.3",
    token=os.getenv("HF_TOKEN")
)

@app.route('/api/generate-itinerary', methods=['POST'])
def generate_itinerary():
    data = request.json
    destination = data.get('destination', '')
    days = data.get('days', '')
    budget = data.get('budget', '')
    companions = data.get('companions', '')

    prompt = (
        f"Create a detailed travel itinerary for {days} days in {destination} "
        f"with a {budget} budget, traveling with {companions}."
    )

    try:
        # ✅ Corrected keyword: max_tokens
        response = client.chat_completion(
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=600,
            temperature=0.7
        )
        return jsonify({"itinerary": response.choices[0].message.content})
    except Exception as e:
        print("InferenceClient error:", e)
        return jsonify({"itinerary": "Error generating itinerary."}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)
