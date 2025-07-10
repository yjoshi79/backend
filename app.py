from flask import Flask, request, jsonify
from flask_cors import CORS
from huggingface_hub import InferenceClient
import os
import traceback  # added for detailed error logging

app = Flask(__name__)
CORS(app)

@app.route('/api/generate-itinerary', methods=['POST'])
def generate_itinerary():
    data = request.json
    destination = data.get('destination', '')
    days = data.get('days', '')
    budget = data.get('budget', '')
    companions = data.get('companions', '')

    prompt = (
        f"Create a detailed travel itinerary for {days} days in {destination} "
        f"for a {budget} budget, traveling with {companions}."
    )

    try:
        token = os.getenv("HF_TOKEN")
        client = InferenceClient(
            model="mistralai/Mistral-7B-Instruct-v0.3",
            token=token
        )

        response = client.text_generation(
            prompt=prompt,
            max_tokens=600,
            temperature=0.7
        )

        return jsonify({"itinerary": response}), 200

    except Exception as e:
        traceback.print_exc()  # <-- This prints full traceback in logs
        print("InferenceClient error:", e)
        return jsonify({"itinerary": "Error generating itinerary"}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
