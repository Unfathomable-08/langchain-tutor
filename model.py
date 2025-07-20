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

# Prompt template for exercise evaluation
exercise_prompt_template = PromptTemplate(
    input_variables=["question", "user_answer"],
    template=(
        "You are an Arabic tutor evaluating a beginner's exercise response. "
        "The exercise question is: {question}\n"
        "The user's Arabic response is: {user_answer}\n"
        "Score the response out of 10 based on relevance, grammar, and clarity. "
        "Provide the score and one short sentence with feedback, suggestion, or praise in English, including one Arabic word or phrase to reinforce learning. "
        "Format the response as: Score: X/10\nFeedback: [Your feedback]"
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

# Function to evaluate Arabic exercise responses
def evaluate_exercise(question, user_answer):
    try:
        # Format the exercise prompt
        prompt = exercise_prompt_template.format(question=question, user_answer=user_answer)
        # Call the conversational API
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=80,
            temperature=0.2  # Lower temperature for more consistent scoring
        )
        result = response.choices[0].message.content
        # Save to memory
        memory.save_context(
            {"question": f"Exercise: {question}, User Answer: {user_answer}"},
            {"answer": result}
        )
        return result
    except Exception as e:
        return f"Error: {str(e)}"