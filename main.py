import logging
from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
from typing import Dict, Any, List
import uuid
import json
from agents.recommendation_engine import get_next_step_recommendations
import pandas as pd
from langchain_core.messages import HumanMessage, AIMessage
import io
from langchain_core.prompts import ChatPromptTemplate
from state.graph_state import MLState
from agents.router import get_routing_decision
from tools.load_dataset_tool import load_dataset_tool, list_available_datasets
from llm.task_identifier_nodes import (
    identify_task_node,
    request_confirmation_node,
    handle_validation_node,
)
from tools.show_data_tool import show_data_tool_logic
from tools.setup_tool import setup_tool
from tools.compare_model_tool import compare_models_tool
from tools.create_model_tool import  create_model_tool
from tools.tune_model_tool import  tune_model_tool
from tools.ensemble_model_tool import ensemble_model_tool
from tools.automl_tool import automl_tool
from tools.leaderboard_tool import leaderboard_tool
from tools.finalize_model_tool import finalize_model_tool
from tools.save_model_tool import save_model_tool
from tools.load_model_tool import load_model_tool
from tools.assign_model_tool import assign_model_tool_logic
from tools.data_analysis_tool import (
    descriptive_statistics_tool_logic,
    missing_values_tool_logic,
    correlation_analysis_tool_logic,
)
from tools.show_model_tool import show_model_tool
from tools.plot_tool import plot_model_tool
from langchain_groq import ChatGroq
from tools.eda_plot_tool import eda_plot_tool_logic
from dotenv import load_dotenv
load_dotenv()

# --- Configure logging and LLM ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
llm = ChatGroq(temperature=0.1, model_name="llama3-8b-8192", model_kwargs={"response_format": {"type": "json_object"}})

# 👈 ADD THIS: A more powerful LLM specifically for routing
router_llm = ChatGroq(temperature=0.0, model_name="llama3-70b-8192", model_kwargs={"response_format": {"type": "json_object"}})


# --- Initialize FastAPI App ---
app = FastAPI(
    title="AutoML Agent API",
    description="An API for interacting with a natural-language-driven ML system.",
)

# --- In-Memory Session Storage ---
sessions: Dict[str, MLState] = {}

# --- Pydantic Models for API Requests/Responses ---
class ChatRequest(BaseModel):
    session_id: str # Session ID is now required for chat
    user_query: str

class ChatResponse(BaseModel):
    session_id: str
    assistant_message: str
    recommendations: List[str] = []

# UPDATE THIS MODEL to expect a single query string
class UploadResponse(BaseModel):
    session_id: str
    filename: str
    message: str
    data_preview: str
    suggested_queries: List[str]


def create_serializable_state_summary(state: MLState) -> Dict[str, Any]:
    """
    Creates a JSON-safe dictionary summary of the MLState, excluding complex objects.
    """
    summary = {
        "data_loaded": state.data is not None,
        "data_shape": state.data.shape if isinstance(state.data, pd.DataFrame) else None,
        "task": state.task,
        "target_column": state.target_column,
        "setup_done": state.setup_done,
        "model_available": any([
            state.model, state.tuned_model, state.final_model, 
            state.best_model, state.ensemble_model
        ]),
        "leaderboard_available": state.leaderboard is not None,
    }
    # Return only the keys that have a value
    return {k: v for k, v in summary.items() if v is not None}

# --- Tool Mapping ---
tool_map = {
    "load_dataset_tool": load_dataset_tool,
    "descriptive_statistics_tool": descriptive_statistics_tool_logic,
    "missing_values_tool":  missing_values_tool_logic,
    "correlation_analysis_tool": correlation_analysis_tool_logic,
    "plot_model_tool": plot_model_tool,
    "setup_tool":  setup_tool,
    "compare_models_tool": compare_models_tool,
    "create_model_tool": create_model_tool,
    "tune_model_tool": tune_model_tool,
    "ensemble_model_tool": ensemble_model_tool,
    "automl_tool": automl_tool,
    "leaderboard_tool": leaderboard_tool,
    "finalize_model_tool": finalize_model_tool,
    "save_model_tool": save_model_tool,
    "load_model_tool": load_model_tool,
    "assign_model_tool": assign_model_tool_logic,
    "show_model_tool": show_model_tool,
    "show_data_tool": show_data_tool_logic,
    "eda_plot_tool": eda_plot_tool_logic,
}

# --- API Endpoints ---

@app.get("/")
def read_root():
    return {"message": "Welcome to the AutoML Agent API."}

# --- API Endpoints ---

@app.post("/upload", response_model=UploadResponse)
async def upload_dataset(file: UploadFile = File(...)):
    """
    Handles CSV file uploads, creates a new session, stores the data,
    and generates a single suggested ML query.
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a CSV file.")
    
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))

        session_id = str(uuid.uuid4())
        state = MLState(input_message="", data=df)
        sessions[session_id] = state

        # --- RESTORED: Generate one suggested query ---
        prompt = ChatPromptTemplate.from_template(
    "Based on these column names and data preview, suggest three concise, relevant natural language questions a user might ask to start a machine learning task. "
    "For example: 'Predict the top speed based on other features' or 'Classify the battery type'. "
    "You MUST respond with a valid JSON object with a single key 'queries' containing a list of exactly 3 question strings. "
    "Do NOT include any introductory text, explanations, or markdown formatting. Only the JSON object.\n\n"
    "Columns: {columns}\nPreview:\n{preview}\n\nJSON Output:"
)
        chain = prompt | llm
        response = chain.invoke({"columns": list(df.columns), "preview": df.head(2).to_string()})
        # The response is now a simple string
        suggested_queries = json.loads(response.content).get("queries", [])

        logger.info(f"New session {session_id} created for file {file.filename}")

        return UploadResponse(
            session_id=session_id,
            filename=file.filename,
            message=f"✅ Dataset '{file.filename}' loaded successfully! Shape: {df.shape}.",
            data_preview=df.head().to_string(),
            suggested_queries=suggested_queries
        )
    except Exception as e:
        logger.error(f"Failed to process uploaded file: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process file: {e}")


@app.post("/chat", response_model=ChatResponse)
def chat_with_agent(request: ChatRequest):
    session_id = request.session_id
    state = sessions.get(session_id)
    if not state:
        raise HTTPException(status_code=404, detail="Session not found.")

    user_query = request.user_query
    updates = {} 
    # --- Onboarding & Main Logic ---
    if state.task is None:
        state.input_message = user_query
        updates = identify_task_node(state)
        # We need to chain another call here for the confirmation request
        temp_state_after_identify = state.model_copy(update=updates)
        updates.update(request_confirmation_node(temp_state_after_identify))
    
    elif state.user_validation_response != "validated":
        updates = handle_validation_node(state.model_copy(update={"user_validation_response": user_query}))
        if "✅" in updates.get("last_output", ""):
            updates["user_validation_response"] = "validated"
        elif "Apologies" in updates.get("last_output", ""):
            updates["user_validation_response"] = None

    else: # Main tool loop
        decision = get_routing_decision(user_query)
        if not decision or "tool_name" not in decision:
            updates["last_output"] = "I'm sorry, I couldn't understand that command."
        else:
            tool_name = decision["tool_name"]
            tool_to_call = tool_map.get(tool_name)
            if not tool_to_call:
                updates["last_output"] = f"Tool '{tool_name}' is not implemented."
            else:
                updates = tool_to_call.invoke({"state": state, "user_query": user_query})

    # --- Step 2: Generate contextual recommendations and update history ---
    last_output = updates.get("last_output", "An action was performed.")
    
    full_message_history = state.messages + [
        HumanMessage(content=user_query),
        AIMessage(content=last_output),
    ]

    # === MODIFICATION START: Pass column names to the engine ===
    temp_state = state.model_copy(update=updates)
    
    # Get column names if data exists, otherwise provide an empty list
    column_names = list(state.data.columns) if state.data is not None else []
    
    recommendations = get_next_step_recommendations(
        state=temp_state,
        last_tool_output=last_output,
        messages=full_message_history,
        column_names=column_names  # <-- Pass the actual column names
    )
    # === MODIFICATION END ===

    updates["recommendations"] = recommendations
    updates["messages"] = full_message_history

    # --- Step 3: Apply all updates and save the new state ---
    state = state.model_copy(update=updates)
    sessions[session_id] = state
    
    # --- Step 4: Return the response ---
    return ChatResponse(
        session_id=session_id,
        assistant_message=state.last_output,
        recommendations=state.recommendations
    )
