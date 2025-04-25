import google.generativeai as genai
from utils.vector_store import collection
import os
from dotenv import load_dotenv

load_dotenv()

# Configure the API key
genai.configure(api_key="AIzaSyB9IHWHbqggP__-hN9304vrJqTnvTDha3c")

class AgriBot:
    def __init__(self):
        system_instruction = """
        You are AgriBot, a friendly, knowledgeable agricultural assistant for Ugandan farmers.
        Your responses should:
        - Be written in simple language, easy to understand by rural farmers.
        - Be structured in clear bullet points.
        - Be specific to Uganda's crops, soil, and climate whenever possible.
        - Never answer questions unrelated to agriculture. Politely redirect.
        - Keep answers short, direct, and practical.
        """

        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            system_instruction=system_instruction
        )

        self.chat = model.start_chat(history=[])
        self.relevance_threshold = 0.65
        self.response_source = "vector_store"

    def start_chat(self):
        self.__init__()

    def get_response(self, query, include_sources=True):
        intent_prompt = f"""
        You are an intent classifier for a Ugandan agricultural chatbot.

        Categorize the user's input into one of the following intents:
        - greeting
        - goodbye
        - help
        - who_are_you
        - ask_agriculture_question
        - unknown

        Respond ONLY with one of the intent labels above. No explanation.

        Input: {query}
        """

        intent_model = genai.GenerativeModel("gemini-1.5-flash")
        intent_result = intent_model.generate_content(intent_prompt).text.strip().lower()

        if intent_result == "greeting":
            self.response_source = "intent_handler"
            return "Hello! I'm **AgriBot**, your farming assistant. Ask me anything about crops, soil, pests, or planting in Uganda."

        elif intent_result == "goodbye":
            self.response_source = "intent_handler"
            return "Goodbye! Wishing you good harvests and healthy crops. Come back anytime you need farming advice."

        elif intent_result == "help":
            self.response_source = "intent_handler"
            return (
                "Here's how I can help:\n"
                "• **Planting Advice:** Learn the best way to plant various Ugandan crops.\n"
                "• **Pest and Disease Control:** Get tips on identifying and managing common agricultural issues.\n"
                "• **Soil Health:** Understand how to improve your soil for better yields.\n"
                "• **Fertilizers:** Find out about different types of fertilizers and their use.\n"
                "• Just type your question to get started!"
            )

        elif intent_result == "who_are_you":
            self.response_source = "intent_handler"
            return "I'm **AgriBot**, your AI-powered assistant for Ugandan farmers. I give simple, practical answers from expert farming resources."

        elif intent_result == "ask_agriculture_question" or intent_result == "unknown":
            results = collection.query(query_texts=[query], n_results=3)

            if results['documents'][0]:
                best_match_distance = results['distances'][0][0] if 'distances' in results else None
                similarity_score = 1 - best_match_distance if best_match_distance is not None else 1.0

                if similarity_score >= self.relevance_threshold:
                    self.response_source = "vector_store"

                    context = "\n\n".join([
                        f"Source: {meta['source']}\nContent: {doc}"
                        for doc, meta in zip(results['documents'][0], results['metadatas'][0])
                    ])

                    prompt = f"""
                    Context:
                    {context}

                    Question: {query}

                    Answer in clear bullet points, where each point is a key step or piece of advice, followed by a concise explanation of that point. Use simple language suitable for Ugandan farmers.

                    Answer:
                    """

                    response = self.chat.send_message(prompt)
                    response_text = response.text
                else:
                    self.response_source = "gemini_fallback"
                    response_text = self._get_gemini_agriculture_response(query)
            else:
                self.response_source = "gemini_fallback"
                response_text = self._get_gemini_agriculture_response(query)

            # Format the response into "Point: Explanation" structure
            formatted_response = ""
            for item in response_text.split("\n"):
                item = item.strip()
                if item.startswith("•"):
                    parts = item[1:].strip().split(":", 1)
                    if len(parts) == 2:
                        point = parts[0].strip()
                        explanation = parts[1].strip()
                        formatted_response += f"**{point}:** {explanation}\n"
                    else:
                        formatted_response += f"• {item[1:].strip()}\n" # Handle cases where explanation might be missing
                elif item:
                    formatted_response += f"{item}\n" # Keep other non-bulleted information

            formatted_response = formatted_response.strip()
            formatted_response = self._limit_words(formatted_response, max_words=250) # Adjusted max words

            if include_sources and self.response_source == "vector_store":
                sources = "\n\nReferences:\n" + "\n".join(
                    [f"- {meta['source']}" for meta in results['metadatas'][0]]
                )
                return f"{formatted_response}\n\n{sources}"

            return formatted_response

        else:
            self.response_source = "fallback"
            return "Sorry, I didn't understand that. Can you rephrase your question or ask me something about farming in Uganda?"

    def _get_gemini_agriculture_response(self, query):
        fallback_model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            system_instruction="""
            You are AgriBot, a specialized agricultural assistant for Ugandan farmers.
            Provide advice focused on:
            - Crop cultivation in Uganda
            - Pest/disease control for local crops
            - Soil health and fertilizers
            - Sustainable agriculture
            - Post-harvest handling

            Answer in clear bullet points, where each point is a key step or piece of advice, followed by a concise explanation of that point. Use simple language suitable for Ugandan farmers.
            """
        )
        fallback_chat = fallback_model.start_chat(history=[])
        response = fallback_chat.send_message(query)
        return response.text

    def _limit_words(self, text, max_words=200): # Keep this for overall length control
        words = text.split()
        if len(words) <= max_words:
            return text
        return " ".join(words[:max_words]) + "..."

    def get_response_source(self):
        return self.response_source

    # def get_response(self, query, include_sources=True):
    #     intent_prompt = f"""
    #     You are an intent classifier for a Ugandan agricultural chatbot. 

    #     Categorize the user's input into one of the following intents:
    #     - greeting
    #     - goodbye
    #     - help
    #     - who_are_you
    #     - ask_agriculture_question
    #     - unknown

    #     Respond ONLY with one of the intent labels above. No explanation.

    #     Input: {query}
    #     """

    #     intent_model = genai.GenerativeModel("gemini-1.5-flash")
    #     intent_result = intent_model.generate_content(intent_prompt).text.strip().lower()

    #     if intent_result == "greeting":
    #         self.response_source = "intent_handler"
    #         return "Hello! I'm **AgriBot**, your farming assistant. Ask me anything about crops, soil, pests, or planting in Uganda."

    #     elif intent_result == "goodbye":
    #         self.response_source = "intent_handler"
    #         return "Goodbye! Wishing you good harvests and healthy crops. Come back anytime you need farming advice."

    #     elif intent_result == "help":
    #         self.response_source = "intent_handler"
    #         return (
    #             "Here's how I can help:\n"
    #             "• Ask how to plant crops like maize, beans, or tomatoes.\n"
    #             "• Get tips on dealing with pests and diseases.\n"
    #             "• Learn about soil health, fertilizers, and climate.\n"
    #             "• Just type your question to get started!"
    #         )

    #     elif intent_result == "who_are_you":
    #         self.response_source = "intent_handler"
    #         return "I'm **AgriBot**, your AI-powered assistant for Ugandan farmers. I give simple, practical answers from expert farming resources."

    #     elif intent_result == "ask_agriculture_question" or intent_result == "unknown":
    #         results = collection.query(query_texts=[query], n_results=3)

    #         if results['documents'][0]:
    #             best_match_distance = results['distances'][0][0] if 'distances' in results else None
    #             similarity_score = 1 - best_match_distance if best_match_distance is not None else 1.0    

    #             if similarity_score >= self.relevance_threshold:
    #                 self.response_source = "vector_store"

    #                 context = "\n\n".join([
    #                     f"Source: {meta['source']}\nContent: {doc}"
    #                     for doc, meta in zip(results['documents'][0], results['metadatas'][0])
    #                 ])

    #                 prompt = f"""
    #                 Context:
    #                 {context}

    #                 Question: {query}

    #                 Answer in clear bullet points, where each point is a key step or piece of advice, followed by a concise explanation of that point. 
    #                 Use simple language suitable for Ugandan farmers.

    #                 Answer:
    #                 """

    #                 response = self.chat.send_message(prompt)
    #                 response_text = response.text
    #             else:
    #                 self.response_source = "gemini_fallback"
    #                 response_text = self._get_gemini_agriculture_response(query)
    #         else:
    #             self.response_source = "gemini_fallback"
    #             response_text = self._get_gemini_agriculture_response(query)

    #         # Structure as bullet points
    #         chunked_response = "\n".join([
    #             f"• {line.strip()}" if not line.strip().startswith("•") else line.strip()
    #             for line in response_text.split("\n") if line.strip()
    #         ])

    #         # ✂️ Limit word count
    #         chunked_response = self._limit_words(chunked_response, max_words=200)

    #         if include_sources and self.response_source == "vector_store":
    #             sources = "\n\nReferences:\n" + "\n".join(
    #                 [f"- {meta['source']}" for meta in results['metadatas'][0]]
    #             )
    #             return f"{chunked_response}\n\n{sources}"

    #         return chunked_response

    #     else:
    #         self.response_source = "fallback"
    #         return "Sorry, I didn't understand that. Can you rephrase your question or ask me something about farming in Uganda?"

    # def _get_gemini_agriculture_response(self, query):
    #     fallback_model = genai.GenerativeModel(
    #         model_name="gemini-1.5-flash",
    #         system_instruction="""
    #         You are AgriBot, a specialized agricultural assistant for Ugandan farmers.
    #         Provide advice focused on:
    #         - Crop cultivation in Uganda
    #         - Pest/disease control for local crops
    #         - Soil health and fertilizers
    #         - Sustainable agriculture
    #         - Post-harvest handling

    #         Use simple bullet points in farmer-friendly language.
    #         """
    #     )
    #     fallback_chat = fallback_model.start_chat(history=[])
    #     response = fallback_chat.send_message(query)
    #     return response.text

    # def _limit_words(self, text, max_words=200):
    #     words = text.split()
    #     if len(words) <= max_words:
    #         return text
    #     return " ".join(words[:max_words]) + "..."

    # def get_response_source(self):
    #     return self.response_source
