# Autonomous AutoML Agent 
Conversational Machine Learning Automation with Streamlit & FastAPI
This project is an interactive AutoML agent that lets users build, analyze, and deploy machine learning models using natural language. It combines a Streamlit frontend with a FastAPI backend, powered by PyCaret and LLMs (Groq Llama3), to automate the entire ML workflow—no coding required.

Features
1.Conversational UI: Chat with the agent to upload data, analyze, preprocess, build models, tune, ensemble, visualize, and deploy—all via natural language.
2.Multi-session Chat: Manage multiple ML sessions and view chat history.
3.Automated Recommendations: Get next-step suggestions based on your workflow and data.
4.EDA & Visualization: Generate plots and data summaries on demand.
5.Model Management: Save, load, and finalize models for deployment.
6.Powered by LLMs: Uses Groq Llama3 for query understanding, tool routing, and parameter extraction.

How It Works
1.Upload a CSV dataset via the Streamlit UI.
2.Chat with the agent to describe your ML goal.
3.The agent identifies the ML task and guides you through setup, analysis, modeling, and evaluation.
4.Visualize results and get actionable recommendations for next steps.
5.Save or deploy models with a single command.

Prerequisites
Python 3.9+
PyCaret
Streamlit
FastAPI
LangChain
Groq API Key (add to .env)
