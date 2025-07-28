import logging
import os
import json
import pandas as pd
import numpy as np

# === MODIFICATION START: Set non-interactive backend ===
import matplotlib
matplotlib.use('Agg')
# === MODIFICATION END ===

import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any

from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from state.graph_state import MLState

logger = logging.getLogger(__name__)
llm = ChatGroq(
    temperature=0.0,
    model_name="llama3-8b-8192",
    model_kwargs={"response_format": {"type": "json_object"}},
)

def _get_llm_eda_plot_params(user_query: str, columns: list) -> Dict[str, Any] | None:
    """Uses an LLM to extract plot type and columns for EDA plots."""
    prompt = ChatPromptTemplate.from_template("""
You are an expert at parsing visualization requests. From the user's query, extract the plot type and the column(s) to be plotted.

Supported plot types: 'histogram', 'boxplot', 'scatter', 'heatmap', 'bar'.
- A 'histogram' or 'boxplot' requires one numeric column ('x').
- A 'bar' plot requires one categorical column ('x').
- A 'scatter' plot requires two numeric columns ('x' and 'y').
- A 'heatmap' does not require specific columns; it uses all numeric columns.

Available columns: {columns}

User Query: {user_query}

JSON Output (e.g., {{"plot_type": "bar", "x": "category_column"}}):
""")
    chain = prompt | llm
    try:
        response = chain.invoke({"columns": columns, "user_query": user_query})
        return json.loads(response.content)
    except Exception as e:
        logger.error(f"LLM call for EDA plot params failed: {e}")
        return None

@tool("eda_plot_tool")
def eda_plot_tool_logic(state: MLState, user_query: str) -> Dict[str, Any]:
    """
    Creates and saves an Exploratory Data Analysis (EDA) plot for the raw dataset.
    Supports 'histogram', 'boxplot', 'scatter', 'heatmap', and 'bar'.
    """
    logger.info("--- Executing EDA Plot Tool ---")
    data = state.data
    if data is None:
        return {"last_output": "❌ No data loaded. Please load a dataset first."}

    params = _get_llm_eda_plot_params(user_query, list(data.columns))
    if not params or not params.get("plot_type"):
        return {"last_output": "❌ Could not determine the plot type. Please specify 'histogram', 'boxplot', 'scatter', 'heatmap', or 'bar'."}

    plot_type = params.get("plot_type")
    x_col = params.get("x")
    y_col = params.get("y")
    
    os.makedirs("plots", exist_ok=True)
    file_path = f"plots/eda_{plot_type}_{x_col or ''}.png"
    
    try:
        plt.figure(figsize=(12, 7))
        if plot_type == 'histogram':
            if not x_col: return {"last_output": "❌ A histogram requires one numeric column."}
            sns.histplot(data=data, x=x_col, kde=True)
            plt.title(f'Distribution of {x_col}')
            
        elif plot_type == 'boxplot':
            if not x_col: return {"last_output": "❌ A boxplot requires one numeric column."}
            sns.boxplot(data=data, x=x_col)
            plt.title(f'Box Plot of {x_col}')

        elif plot_type == 'bar':
            if not x_col: return {"last_output": "❌ A bar plot requires one categorical column."}
            sns.countplot(data=data, x=x_col, order = data[x_col].value_counts().index)
            plt.title(f'Frequency of Categories in {x_col}')
            plt.xticks(rotation=45, ha='right')
            
        elif plot_type == 'scatter':
            if not x_col or not y_col: return {"last_output": "❌ A scatter plot requires two numeric columns."}
            sns.scatterplot(data=data, x=x_col, y=y_col)
            plt.title(f'Scatter Plot of {x_col} vs {y_col}')
            
        elif plot_type == 'heatmap':
            numeric_data = data.select_dtypes(include=np.number)
            if numeric_data.shape[1] < 2: return {"last_output": "❌ A heatmap requires at least two numeric columns."}
            sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
            plt.title('Correlation Heatmap')
        else:
            return {"last_output": f"❌ Plot type '{plot_type}' is not supported."}

        plt.tight_layout()
        plt.savefig(file_path, dpi=150)
        plt.close()
        
        last_output = f"✅ EDA plot '{plot_type}' generated successfully. You can view it at: `{file_path}`"
        return {"plot_path": file_path, "last_output": last_output}

    except Exception as e:
        last_output = f"❌ Failed to generate EDA plot: {e}"
        return {"last_output": last_output}