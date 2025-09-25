from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables.graph import MermaidDrawMethod
from langchain_groq import ChatGroq
from typing import TypedDict, Dict
import gradio as gr
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")


# Initialize Groq LLM
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, groq_api_key=groq_api_key)

# Define State
class ChatState(TypedDict):
    question: str
    category: str
    emotion:  str
    response: str

# --- Categorization Node ---
def categorize_node(state: ChatState) -> ChatState:
    prompt = ChatPromptTemplate.from_template(
        "You are a helpful assistant that classifies customer questions "
        "into one of the categories: Billing, Technical Support, General Inquiry, or Unknown.\n\n"
        "Question: {question}\n\n"
        "Return only the category."
    )
    chain = prompt | llm
    category = chain.invoke({"question": state["question"]}).content.strip()

    return {**state, "category": category}


# --- Emotion Detection Node ---
def emotion_node(state: ChatState) -> ChatState:
    prompt = ChatPromptTemplate.from_template(
        "You are an assistant that analyzes customer questions and detects the emotion.\n"
        "Possible emotions: Positive, Frustrated, Neutral, Not Happy, Negative\n\n"
        "Question: {question}\n\n"
        "Return only the detected emotion."
    )
    chain = prompt | llm
    emotion = chain.invoke({"question": state["question"]}).content.strip()

    return {**state, "emotion": emotion}


# --- Billing Node ---
def billing_node(state: ChatState) -> ChatState:
    prompt = ChatPromptTemplate.from_template(
        "You are a customer support assistant.\n\n"
        "The customer asked a billing-related question:\n"
        "{question}\n\n"
        "Provide a short, helpful, and friendly billing response."
    )
    chain = prompt | llm
    response = chain.invoke({"question": state["question"]}).content.strip()

    return {**state, "response": response}


# --- Technical Support Node ---
def technical_node(state: ChatState) -> ChatState:
    prompt = ChatPromptTemplate.from_template(
        "You are a customer support assistant.\n\n"
        "The customer asked a technical support question:\n"
        "{question}\n\n"
        "Provide a short, helpful, and friendly technical support response."
    )
    chain = prompt | llm
    response = chain.invoke({"question": state["question"]}).content.strip()

    return {**state, "response": response}


# --- General Inquiry Node ---
def inquiry_node(state: ChatState) -> ChatState:
    prompt = ChatPromptTemplate.from_template(
        "You are a customer support assistant.\n\n"
        "The customer asked a general inquiry question:\n"
        "{question}\n\n"
        "Provide a short, helpful, and friendly response."
    )
    chain = prompt | llm
    response = chain.invoke({"question": state["question"]}).content.strip()

    return {**state, "response": response}


# --- Escalation Node ---
def escalate_node(state: ChatState) -> ChatState:
    return {**state, "response": "This question has been escalated to a human representative"}

# --- Routing Function ---
def router_node(state: ChatState) -> str:
    """
    - Escalates if user is frustrated/angry
    - Otherwise routes based on category
    """
    # Escalation check
    if state["emotion"].lower() in ["frustrated", "angry", "upset", "not happy"]:
        return "escalate"

    # Category-based routing
    if state["category"] == "Billing":
        return "billing"
    elif state["category"] == "Technical Support":
        return "technical"
    elif state["category"] == "General Inquiry":
        return "inquiry"
    else:
        return "inquiry"  # default fallback


# --- Graph Definition ---
graph = StateGraph(ChatState)

# Add nodes
graph.add_node("categorize", categorize_node)
graph.add_node("emotion", emotion_node)
graph.add_node("billing", billing_node)
graph.add_node("technical", technical_node)
graph.add_node("inquiry", inquiry_node)
graph.add_node("escalate", escalate_node)

# Add edges
graph.add_edge("categorize", "emotion")
graph.add_conditional_edges("emotion", router_node, ["billing", "technical", "inquiry", "escalate"])

# Set entry point
graph.set_entry_point("categorize")

# Compile graph
app = graph.compile()


# --- RUN FUNCTION ---
def run_chatbot(question: str):
    # initial state
    initial_state = {
        "question": question,
        "category": "",
        "emotion": "",
        "response": ""
    }

    # run graph
    result = app.invoke(initial_state)

    # always return consistent fields
    return {
        "question": result.get("question", ""),
        "category": result.get("category", ""),
        "emotion": result.get("emotion", ""),
        "response": result.get("response", "")
    }

# Gradio interface

def interface(question: str):
    results= run_chatbot(question)

    return (
        f"Question: {results['question']}\n"
        f"Category: {results['category']}\n"
        f"Emotion: {results['emotion']}\n\n"
        f"Response: {results['response']}"
    )


with gr.Blocks() as demo:
    gr.Markdown("## 🤖 Customer Support Chatbot")

    with gr.Row():
        with gr.Column(scale=3):
            question = gr.Textbox(label="Your Question", placeholder="Type your question here...")
            submit = gr.Button("Ask")
        with gr.Column(scale=3):
            answer = gr.Textbox(label="Bot Response", interactive=False, lines=6, max_lines=10)

    # On button click → run chatbot
    submit.click(fn=interface, inputs=question, outputs=answer)

# Launch the app
if __name__ == "__main__":
    demo.launch()