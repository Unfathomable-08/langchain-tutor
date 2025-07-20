from flask import Flask, request, jsonify
from flask_cors import CORS
from model import get_response, evaluate_exercise

app = Flask(__name__)
CORS(app)

# Route to predict
@app.route('/predict', methods=['POST'])
def answer():
    response = None
    error = None

    if request.method == "POST":
        # Get the question from the form
        data = request.get_json()
        question = data.get("question")
        if question:
            # Call the get_response function from model.py
            result = get_response(question)
            if result.startswith("Error:"):
                error = result
            else:
                response = result
        else:
            error = "Please enter a question."

    return jsonify({'answer': response})

# Route to evaluate exercise
@app.route('/evaluate', methods=['POST'])
def evaluate():
    response = None
    error = None

    if request.method == "POST":
        # Get the question and user answer from the form
        data = request.get_json()
        question = data.get("question")
        user_answer = data.get("user_answer")
        if question and user_answer:
            # Call the evaluate_exercise function from model.py
            result = evaluate_exercise(question, user_answer)
            if result.startswith("Error:"):
                error = result
            else:
                response = result
        else:
            error = "Please provide both a question and an answer."

    return jsonify({'answer': response, 'error': error})

# Default route
@app.route('/', methods=['GET'])
def res():
    return jsonify({'answer': "hit /predict with post api"})

if __name__ == '__main__':
    app.run(debug=True)