import google.generativeai as genai
from utils.vector_store import collection
import os
from dotenv import load_dotenv
import re


load_dotenv()

# Configure the API key
genai.configure(api_key="AIzaSyB9IHWHbqggP__-hN9304vrJqTnvTDha3c")

class AgriBot:
    def __init__(self):
        self.chat = genai.GenerativeModel("gemini-1.5-flash").start_chat(history=[])

    def start_chat(self):
        self.chat = genai.GenerativeModel("gemini-1.5-flash").start_chat(history=[])

    def get_response(self, query, include_sources=True):
        # Step 1: Use Gemini to detect intent
        intent_prompt = f"""
        You are an intent classifier for a Ugandan agricultural chatbot. 

        Categorize the user's input into one of the following intents:
        - greeting
        - goodbye
        - help
        - who_are_you
        - ask_agriculture_question (if it's a farming or agriculture-related question)
        - unknown (if you’re not sure)

        Respond ONLY with one of the intent labels above. No explanation.

        Input: {query}
        """

        intent_model = genai.GenerativeModel("gemini-1.5-flash")
        intent_result = intent_model.generate_content(intent_prompt).text.strip().lower()

        # Step 2: Respond based on intent
        if intent_result == "greeting":
            return "Hello! I’m **AgriBot**, your farming assistant. Ask me anything about crops, soil, pests, or planting in Uganda."

        elif intent_result == "goodbye":
            return "Goodbye! Wishing you good harvests and healthy crops. Come back anytime you need farming advice."

        elif intent_result == "help":
            return (
                "Here’s how I can help:\n"
                "• Ask how to plant crops like maize, beans, or tomatoes.\n"
                "• Get tips on dealing with pests and diseases.\n"
                "• Learn about soil health, fertilizers, and climate.\n"
                "• Just type your question to get started!"
            )

        elif intent_result == "who_are_you":
            return "I’m **AgriBot**, your AI-powered assistant for Ugandan farmers. I give simple, practical answers from expert farming resources."

        elif intent_result == "ask_agriculture_question":
            # Step 3: Proceed with vector search
            results = collection.query(query_texts=[query], n_results=3)

            # Step 4: Create context from retrieved documents
            context = "\n\n".join([
                f"Source: {meta['source']}\nContent: {doc}"
                for doc, meta in zip(results['documents'][0], results['metadatas'][0])
            ])

            prompt = f"""
            You are AgriBot, an agricultural assistant for Ugandan farmers.
            Provide short, simple, and practical advice in manageable bullet points based on the context below.
            Avoid complex language. Be direct and farmer-friendly.

            Context:
            {context}

            Question: {query}

            Answer:
            """

            response = self.chat.send_message(prompt)
            response_text = response.text

            # Clean and format bullet points
            chunks = response_text.split("\n")
            chunked_response = "\n".join([
                f"• {line.strip().lstrip('*').strip()}"
                for line in chunks if line.strip()
            ])

            # Add references if needed
            if include_sources:
                sources = "\n\nReferences:\n" + "\n".join(
                    [f"- {meta['source']}" for meta in results['metadatas'][0]]
                )
                return f"{chunked_response}\n\n{sources}"

            return chunked_response

        else:
            return "Sorry, I didn't understand that. Can you rephrase your question or ask me something about farming in Uganda?"
