import streamlit as st
import requests
import pandas as pd
import re
from io import StringIO
import os
import uuid

# --- Configuration ---
API_URL = "http://127.0.0.1:8000"

# --- Page Setup ---
st.set_page_config(page_title="AutoML Agent", page_icon="🤖", layout="wide")

# --- Session State Initialization ---
if "view" not in st.session_state:
    st.session_state.view = "landing"
if "chats" not in st.session_state:
    st.session_state.chats = {}
if "active_chat_id" not in st.session_state:
    st.session_state.active_chat_id = None


# --- Helper Functions ---
def parse_assistant_response(response_text):
    """Parses the assistant's response for DataFrames or image paths."""
    plot_path = None; dataframe = None
    plot_path_match = re.search(r"`(plots/[^`]+\.png)`", response_text)
    if plot_path_match and os.path.exists(plot_path_match.group(1)):
        plot_path = plot_path_match.group(1)
        response_text = re.sub(r"`(plots/[^`]+\.png)`", "", response_text)
    dataframe_match = re.search(r"```\n(.*?)```", response_text, re.DOTALL)
    if dataframe_match:
        df_string = dataframe_match.group(1)
        try:
            df = pd.read_csv(StringIO(df_string), sep='\s{2,}', engine='python')
            dataframe = df
            response_text = re.sub(r"```.*```", "\n*Result displayed below.*", response_text, flags=re.DOTALL)
        except Exception: pass
    return response_text.strip(), dataframe, plot_path

def switch_chat(session_id):
    """Sets the active chat session."""
    st.session_state.active_chat_id = session_id


def start_new_chat():
    session_id = str(uuid.uuid4())
    st.session_state.chats[session_id] = {
        "title": "New Chat", "messages": [], "chat_active": False, "plot_gallery": []
    }
    st.session_state.active_chat_id = session_id
    st.session_state.view = "chat"

def send_message(user_query, chat_session):
    if not user_query: return
    chat_session["messages"].append({"role": "user", "content": user_query})
    try:
        payload = {"session_id": st.session_state.active_chat_id, "user_query": user_query}
        response = requests.post(f"{API_URL}/chat", json=payload)
        response.raise_for_status()
        data = response.json()
        text_res, df_res, plot_res = parse_assistant_response(data["assistant_message"])
        assistant_response = {"role": "assistant", "content": text_res}
        if df_res is not None: assistant_response["dataframe"] = df_res
        if plot_res is not None:
            assistant_response["plot_path"] = plot_res
            chat_session["plot_gallery"].append(plot_res)
        if data.get("recommendations"): assistant_response["recommendations"] = data["recommendations"]
        chat_session["messages"].append(assistant_response)
    except requests.exceptions.RequestException as e:
        chat_session["messages"].append({"role": "assistant", "content": f"Error: {e}"})


# --- UI Views ---

def landing_page():
    st.title("Welcome to the Autonomous AutoML Agent 🤖")
    st.header("Build Machine Learning Models with Natural Language")
    st.markdown("""
    This intelligent agent helps you go from raw data to a trained model by simply talking to it.
    - **Upload your dataset** to start a new session.
    - **Analyze your data** with commands like "describe the data" or "check for missing values".
    - **Build models** by saying "compare models" or "create a random forest".
    - **Visualize results** with queries like "plot feature importance".
    """)
    if st.button("Start a New Chat Session", type="primary", use_container_width=True):
        start_new_chat()
        st.rerun()

def chat_page():
    active_chat = st.session_state.chats[st.session_state.active_chat_id]

    # --- Sidebar for Session Control & History ---
    with st.sidebar:
        st.title("AutoML Agent")
        if st.button("➕ New Chat", use_container_width=True):
            start_new_chat(); st.rerun()
        st.markdown("---")
        st.markdown("### Chat History")
        for session_id, chat_data in st.session_state.chats.items():
            if st.button(chat_data["title"], key=session_id, use_container_width=True):
                st.session_state.active_chat_id = session_id; st.rerun()
        st.markdown("---")
        # Plot Gallery
        if active_chat["plot_gallery"]:
            with st.expander("🖼️ Plot Gallery"):
                for plot_path in active_chat["plot_gallery"]:
                    st.image(plot_path)

    # --- Main Chat Interface ---
    st.header(f"Conversation: {active_chat['title']}")
    
    if not active_chat["chat_active"]:
        uploaded_file = st.file_uploader("Upload your CSV Dataset to begin this chat", type="csv")
        if uploaded_file:
            with st.spinner("Uploading and analyzing dataset..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
                    response = requests.post(f"{API_URL}/upload", files=files)
                    response.raise_for_status()
                    data = response.json()
                    new_session_id = data["session_id"]
                    st.session_state.chats[new_session_id] = st.session_state.chats.pop(st.session_state.active_chat_id)
                    st.session_state.active_chat_id = new_session_id
                    active_chat = st.session_state.chats[new_session_id]
                    active_chat["title"] = uploaded_file.name
                    active_chat["chat_active"] = True
                    full_welcome_message = f"{data['message']}\n\n**Data Preview:**\n```\n{data['data_preview']}\n```"
                    text_res, df_res, _ = parse_assistant_response(full_welcome_message)
                    initial_message = {"role": "assistant", "content": text_res}
                    if df_res is not None: initial_message["dataframe"] = df_res
                    if data.get("suggested_queries"):
                        initial_message["recommendations"] = data["suggested_queries"]
                    active_chat["messages"].append(initial_message)
                    st.rerun()
                except requests.exceptions.RequestException as e:
                    st.error(f"Error uploading file: {e}")
    else:
        # Display chat history
        for message in active_chat["messages"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message.get("dataframe") is not None: st.dataframe(message["dataframe"])
                if message.get("plot_path") is not None: st.image(message["plot_path"])

        # Display recommendation buttons
        if active_chat["messages"] and active_chat["messages"][-1]["role"] == "assistant":
            last_message = active_chat["messages"][-1]
            if "recommendations" in last_message:
                st.markdown("**Suggestions:**")
                # UPDATE: Display suggestions as single buttons
                cols = st.columns(len(last_message["recommendations"]))
                for j, rec in enumerate(last_message["recommendations"]):
                    if cols[j].button(rec, key=f"rec_{st.session_state.active_chat_id}_{j}"):
                        send_message(rec, active_chat); st.rerun()
        
        if prompt := st.chat_input("Describe your ML goal or next step..."):
            send_message(prompt, active_chat); st.rerun()


# --- View Router ---
if st.session_state.view == "landing":
    landing_page()
else:
    chat_page()
