# Customer Support [Chatbot](https://huggingface.co/spaces/Fatymahh/customer_support_)

This project is a simple customer support chatbot built using:

- LangChain
 + LangGraph

- Groq LLM
 (Llama 3.3 model)

- Gradio
 for the interactive chat interface

# The bot can:

Classify customer questions into categories: Billing, Technical Support, General Inquiry, Unknown

Detect emotions (Positive, Frustrated, Neutral, Happy)

Generate responses depending on category

Escalate to a human agent if frustration is detected

# Features 🚀

Categorization node for question type.

- Emotion detection node.

- Category-specific response nodes:

- Billing

- Technical Support

- General Inquiry

- Escalation node for frustrated customers.

- Interactive chat-style Gradio UI.

#  Installation 🛠️

Clone the repo

git clone https://github.com/yourusername/customer-support-chatbot.git
cd customer-support-chatbot


Create and activate virtual environment

python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows


Install dependencies

pip install -r requirements.txt

Example requirements.txt
langgraph
langchain
langchain-core
langchain-groq
python-dotenv
gradio

# Environment Variables 🔑 

Create a .env file in the project root and add your Groq API key:

GROQ_API_KEY=your_api_key_here

# Run the Chatbot ▶️ 
python app.py


This will launch a Gradio chat interface in your browser at:
👉 http://127.0.0.1:7860
