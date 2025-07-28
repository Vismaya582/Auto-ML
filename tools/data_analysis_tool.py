import logging
import pandas as pd
import numpy as np
from typing import Dict, Any

from langchain_core.tools import tool
from state.graph_state import MLState
# === MODIFICATION START: Removed unused import ===
# from agents.recommendation_engine import get_next_step_recommendations
# === MODIFICATION END ===

# Import the specific pull functions from each PyCaret module
from pycaret.classification import pull as classification_pull
from pycaret.regression import pull as regression_pull
from pycaret.clustering import pull as clustering_pull
from pycaret.anomaly import pull as anomaly_pull

logger = logging.getLogger(__name__)

def _get_active_dataframe(state: MLState) -> pd.DataFrame | None:
    """
    Returns the transformed dataframe if setup is complete, otherwise returns
    the original raw dataframe.
    """
    if state.setup_done:
        logger.info("Setup is complete. Pulling transformed data for analysis.")
        pull_map = {
            "classification": classification_pull, "regression": regression_pull,
            "clustering": clustering_pull, "anomaly": anomaly_pull,
        }
        pull_function = pull_map.get(state.task)
        if pull_function: return pull_function()
    logger.info("Setup not complete. Using original raw data for analysis.")
    return state.data

@tool("descriptive_statistics_tool")
def descriptive_statistics_tool_logic(state: MLState, user_query: str = None) -> Dict[str, Any]:
    """
    Calculates and displays descriptive statistics for the current dataset.
    If setup has been run, it analyzes the transformed data.
    """
    logger.info("--- Executing Descriptive Statistics Tool ---")
    data = _get_active_dataframe(state)
    if data is None: return {"last_output": "❌ No data available for analysis."}
    try:
        stats = data.describe(include='all').to_string()
        last_output = f"📊 Descriptive Statistics:\n\n```\n{stats}\n```\n"
        # === MODIFICATION START: Removed recommendation logic ===
        return {"last_output": last_output}
        # === MODIFICATION END ===
    except Exception as e:
        last_output = f"❌ An error occurred while generating statistics: {e}"
        # === MODIFICATION START: Removed recommendation logic ===
        return {"last_output": last_output}
        # === MODIFICATION END ===

@tool("missing_values_tool")
def missing_values_tool_logic(state: MLState, user_query: str = None) -> Dict[str, Any]:
    """
    Analyzes and reports missing values for the current dataset.
    If setup has been run, it analyzes the transformed data.
    """
    logger.info("--- Executing Missing Values Tool ---")
    data = _get_active_dataframe(state)
    if data is None: return {"last_output": "❌ No data available for analysis."}
    try:
        missing_counts = data.isnull().sum()
        missing_percentage = (missing_counts / len(data)) * 100
        missing_df = pd.DataFrame({'Missing Values': missing_counts, 'Percentage (%)': missing_percentage})
        missing_df = missing_df[missing_df['Missing Values'] > 0]
        
        if missing_df.empty:
            last_output = "✅ No missing values were found in the dataset."
        else:
            last_output = f"🔍 Missing Values Analysis:\n\n```\n{missing_df.to_string()}\n```\n"
        # === MODIFICATION START: Removed recommendation logic ===
        return {"last_output": last_output}
        # === MODIFICATION END ===
    except Exception as e:
        last_output = f"❌ An error occurred while analyzing missing values: {e}"
        # === MODIFICATION START: Removed recommendation logic ===
        return {"last_output": last_output}
        # === MODIFICATION END ===

@tool("correlation_analysis_tool")
def correlation_analysis_tool_logic(state: MLState, user_query: str = None) -> Dict[str, Any]:
    """
    Computes the correlation matrix for numeric variables and identifies pairs
    with a high correlation coefficient (absolute value > 0.8).
    """
    logger.info("--- Executing Correlation Analysis Tool ---")
    data = _get_active_dataframe(state)
    if data is None: return {"last_output": "❌ No data available for analysis."}
    try:
        numeric_data = data.select_dtypes(include=np.number)
        if len(numeric_data.columns) < 2: return {"last_output": "Not enough numeric columns to compute correlations."}
        
        corr_matrix = numeric_data.corr()
        high_corrs = []
        # Find highly correlated pairs
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.8:
                    high_corrs.append(f"- **{corr_matrix.columns[i]}** and **{corr_matrix.columns[j]}**: `{corr_matrix.iloc[i, j]:.2f}`")
        
        high_corr_summary = "\n".join(high_corrs) if high_corrs else "No highly correlated pairs found."
        last_output = f"### 🔗 High Correlation Pairs (>0.8)\n{high_corr_summary}\n\n### Full Correlation Matrix\n\n```\n{corr_matrix.to_string()}\n```\n"
        # === MODIFICATION START: Removed recommendation logic ===
        return {"last_output": last_output}
        # === MODIFICATION END ===
    except Exception as e:
        last_output = f"❌ An error occurred during correlation analysis: {e}"
        # === MODIFICATION START: Removed recommendation logic ===
        return {"last_output": last_output}
        # === MODIFICATION END ===