import logging
import pandas as pd
from typing import Dict, Any

from langchain_core.tools import tool
from state.graph_state import MLState

# Import the specific pull functions from each PyCaret module
from pycaret.classification import pull as classification_pull
from pycaret.regression import pull as regression_pull
from pycaret.clustering import pull as clustering_pull
from pycaret.anomaly import pull as anomaly_pull

# --- Configure logging ---
logger = logging.getLogger(__name__)

# --- Helper function to get the correct data (raw vs. transformed) ---
def _get_active_dataframe(state: MLState) -> pd.DataFrame | None:
    """
    Returns the transformed dataframe if setup is complete, otherwise returns
    the original raw dataframe.
    """
    if state.setup_done:
        logger.info("Setup is complete. Pulling transformed data.")
        pull_map = {
            "classification": classification_pull,
            "regression": regression_pull,
            "clustering": clustering_pull,
            "anomaly": anomaly_pull,
        }
        pull_function = pull_map.get(state.task)
        if pull_function:
            # The pull() function with no arguments returns the transformed dataset
            return pull_function()
    
    logger.info("Setup not complete. Using original raw data.")
    return state.data

@tool("show_data_tool")
def show_data_tool_logic(state: MLState, user_query: str = None) -> Dict[str, Any]:
    """
    Displays a preview of the current active dataset. If setup has been run,
    it shows the transformed data; otherwise, it shows the original raw data.
    
    Args:
        state (MLState): The current state of the ML pipeline.
        user_query (str): The user's query (not used but required for consistency).

    Returns:
        Dict[str, Any]: A dictionary containing the `last_output` with the
                        data preview.
    """
    logger.info("--- Executing Show Data Tool ---")
    
    active_data = _get_active_dataframe(state)
    
    if active_data is None:
        return {"last_output": "❌ No data available to show."}

    try:
        # Determine if the data is raw or transformed for the message
        data_type = "Transformed Data" if state.setup_done else "Original Data"
        
        preview = active_data.head().to_string()
        
        return {
            "last_output": f"**Displaying a preview of the {data_type}:**\n```\n{preview}\n```"
        }
    except Exception as e:
        logger.error(f"Failed to display data: {e}", exc_info=True)
        return {"last_output": f"❌ An error occurred while trying to display the data: {e}"}
