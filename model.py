from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from langchain.memory import ConversationBufferMemory
import os

# Load API key from .env
load_dotenv()
api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Set up the InferenceClient
client = InferenceClient(model="mistralai/Mixtral-8x7B-Instruct-v0.1", token=api_token)

# Initialize memory to store conversation history
memory = ConversationBufferMemory(return_messages=True)

# Prompt template
prompt_template = PromptTemplate(
    input_variables=["history", "question"],
    template=(
        "You are an Arabic tutor for the Fluentyx platform helping beginners. "
        "Use the conversation history for context. Keep your answer mostly in English if not asked for arabic, "
        "but include a little Arabic to support learning (like key words or phrases). "
        "Stay friendly, clear, and keep it under 50–70 words.\n\n"
        "Conversation History:\n{history}\n\n"
        "Current Question: {question}\nAnswer:"
    )
)

# Function to get model response
def get_response(question):
    try:
        # Get conversation history
        history = memory.load_memory_variables({})["history"]
        # Format the prompt
        prompt = prompt_template.format(history=history, question=question)
        # Call the conversational API
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=120,
            temperature=0.5
        )
        answer = response.choices[0].message.content
        # Save to memory
        memory.save_context({"question": question}, {"answer": answer})

        return answer
    except Exception as e:
        return f"Error: {str(e)}"