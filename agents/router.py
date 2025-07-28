import logging
import json
from typing import Dict, Any, List

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# --- Configure logging and LLM ---
logger = logging.getLogger(__name__)
llm = ChatGroq(
    temperature=0.0,
    model_name="llama3-70b-8192",
    model_kwargs={"response_format": {"type": "json_object"}},
)

# --- (The prompt remains the same as our last version) ---
ROUTER_PROMPT_TEMPLATE = """
You are an expert at routing a user's natural language query to the single most appropriate tool. Your only job is to analyze the query and select one tool from the list below.

**CRITICAL INSTRUCTION: Distinguish between ANALYSIS and ACTION.**
- If the user wants to *see*, *check*, *find*, or *describe* something, use an ANALYSIS tool.
- If the user wants to *change*, *fix*, *fill*, or *preprocess* the data, use an ACTION tool like `setup_tool`.

Based on the user's intent, select one of the following tools:

**1. Data Exploration & EDA Visualization:**
- `show_data_tool`: To display a text preview of the current data.
- `descriptive_statistics_tool`: For summary statistics.
- `missing_values_tool`: To check for missing values.
- `correlation_analysis_tool`: For correlation analysis.
- `eda_plot_tool`: For general EDA plots of the raw data.

**2. Data Preprocessing (ACTION):**
- `setup_tool`: To handle data issues or perform preprocessing.

**3. Modeling & AutoML (ACTION):**
- `compare_models_tool`, `automl_tool`, `create_model_tool`, `tune_model_tool`, `ensemble_model_tool`

**4. Model Inspection & Visualization:**
- `leaderboard_tool`, `show_model_tool`, `assign_model_tool`, `plot_model_tool`

**5. Model Persistence (ACTION):**
- `finalize_model_tool`, `save_model_tool`, `load_model_tool`

---
User Query: {user_query}

JSON Output (must contain the key "tool_name"):
"""

# --- The Router Function (Updated) ---
def get_routing_decision(user_query: str) -> Dict[str, Any] | None:
    """
    Uses the LLM to decide which tool to call based on the user's query.
    """
    logger.info(f"Routing query: '{user_query}'")
    
    prompt = ChatPromptTemplate.from_template(ROUTER_PROMPT_TEMPLATE)
    chain = prompt | llm
    
    try:
        response = chain.invoke({"user_query": user_query})
        decision = json.loads(response.content)
        
        # --- KEY FIX: Check for 'tool' OR 'tool_name' ---
        tool_name = decision.get("tool_name") or decision.get("tool")
        
        if not tool_name:
            logger.warning(f"Router LLM failed to return a valid tool name. Response: {decision}")
            return None

        logger.info(f"Router decision: {tool_name}")
        
        # Standardize the output to always use 'tool_name' for the rest of our code
        return {
            "tool_name": tool_name,
            "user_query": user_query
        }
        
    except Exception as e:
        logger.error(f"Router LLM call failed: {e}")
        return None
