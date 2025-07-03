from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
import os

# Load API key from .env
load_dotenv()
api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Set up the InferenceClient
client = InferenceClient(model="mistralai/Mixtral-8x7B-Instruct-v0.1", token=api_token)

# Prompt template
prompt_template = PromptTemplate(
    input_variables=["question"],
    template="You are an Arabic tutor. Explain clearly:\nQuestion: {question}\nAnswer:"
)

# Format the prompt
question = "What does the word مدرسة mean?"
prompt = prompt_template.format(question=question)

# Call the conversational API
response = client.chat.completions.create(
    messages=[{"role": "user", "content": prompt}],
    max_tokens=100,
    temperature=0.3
)

# Print the response
print(response.choices[0].message.content)