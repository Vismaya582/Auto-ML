import logging
import json
from typing import Dict, Any, List

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage
from state.graph_state import MLState
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)
llm = ChatGroq(
    temperature=0.1,
    model_name="llama3-8b-8192",
    model_kwargs={"response_format": {"type": "json_object"}},
)

def _summarize_ml_state_for_llm(state: MLState) -> str:
    """Creates a summarized JSON string of the MLState for the LLM."""
    summary = {
        "data_loaded": state.data is not None,
        "task": state.task,
        "target_column": state.target_column,
        "setup_done": state.setup_done,
        "model_available": any([
            state.model, state.tuned_model, state.final_model, 
            state.best_model, state.ensemble_model
        ]),
        "leaderboard_available": state.leaderboard is not None,
    }
    return json.dumps({k: v for k, v in summary.items() if v}, indent=2)

def _format_messages_for_llm(messages: List[BaseMessage]) -> str:
    """Formats the last few turns of conversation history for the LLM prompt."""
    if not messages:
        return "No conversation history yet."
    return "\n".join([f"- {msg.type}: {msg.content}" for msg in messages[-4:]])

# === MODIFICATION START: Updated function signature and prompt ===
def get_next_step_recommendations(
    state: MLState, 
    last_tool_output: str, 
    messages: List[BaseMessage],
    column_names: List[str]
) -> List[str]:
    """
    Uses an LLM to generate a list of 3 example queries for the user's next step,
    strictly based on the provided context.
    """
    logger.info("Generating query-based recommendations with history...")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a precise data science assistant. Your job is to suggest the next logical steps in a machine learning workflow by providing 3 actionable user queries.

**Your Rules:**
1.  **BE STRICT:** Your suggestions MUST be grounded in the provided context. Do NOT hallucinate or invent column names, plot types, or model names.
2.  **USE PROVIDED COLUMNS:** If suggesting a plot or analysis on a column, you MUST use a column name from the `Available Columns` list.
3.  **STAY RELEVANT:** Suggestions must be a logical continuation of the `Recent Conversation` and the `Last Action Result`.
4.  **BE CONCISE:** Each recommendation must be a short, direct command a user would type.
5.  **OUTPUT FORMAT:** You must provide a JSON object with a single key "recommendations" containing a list of exactly 3 string recommendations.

Example Output:
{{
  "recommendations": [
    "plot a histogram of the 'age' column",
    "check for missing values",
    "show me the correlation matrix"
  ]
}}
"""),
        ("user", """
**Available Columns:**
`{column_names}`

**Recent Conversation:**
{history}

**Current Project State:**
```json
{state_summary}
Last Action Result:
{last_output}

Based only on the information above, what are the most relevant next steps?

Your JSON Output:
""")
])

    chain = prompt | llm

    try:
        response = chain.invoke({
            "column_names": column_names,
            "history": _format_messages_for_llm(messages),
            "state_summary": _summarize_ml_state_for_llm(state),
            "last_output": last_tool_output
        })
        decision = json.loads(response.content)
    
        if "recommendations" in decision and isinstance(decision["recommendations"], list):
            logger.info(f"Generated recommendations: {decision['recommendations']}")
            return decision["recommendations"]
        return []
    except Exception as e:
        logger.error(f"Recommendation engine failed: {e}")
        return []