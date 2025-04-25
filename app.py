from flask import Flask, request, jsonify, render_template
from chatbot.chatbot import AgriBot
import os
import logging

# Set up logging
logging.basicConfig(
    filename='agribot.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

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
    
    # Log the source of the response for monitoring
    response_source = bot.get_response_source()
    logging.info(f"Query: '{query}' | Response source: {response_source}")
    
    # Return both the response and its source
    return jsonify({
        "response": response,
        "source": response_source
    })

if __name__ == '__main__':
    app.run(debug=True)