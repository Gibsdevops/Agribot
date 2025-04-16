from google import genai


client = genai.Client(api_key="AIzaSyB9IHWHbqggP__-hN9304vrJqTnvTDha3c")

response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents="Explain how AI works",
)

print(response.text)