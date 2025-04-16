from flask import Flask, request, jsonify, render_template
from chatbot.chatbot import AgriBot
import os

app = Flask(__name__)
bot = AgriBot()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/start_chat', methods=['POST'])
def start_chat():
    # Starts a new chat session using the AgriBot instance
    bot.start_chat()
    return jsonify({"status": "Chat started"})

@app.route('/chat', methods=['POST'])
def chat():
    # Retrieves the query from the request
    query = request.json.get('query')
    if not query:
        return jsonify({"error": "No query provided"}), 400  # Handle missing query

    # Gets the response from AgriBot
    response = bot.get_response(query)
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True)
