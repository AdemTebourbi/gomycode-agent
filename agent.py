import math
import logging
import os
import re
import requests
from operator import add
from typing import TypedDict, Optional, Annotated

from langgraph.graph import StateGraph, END, START
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agent")


# ─── State ───────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    query: str
    tool_name: Optional[str]
    tool_input: Optional[str]  # The prepared input for the tool (expression or search query)
    tool_result: Optional[str]
    tool_error: Optional[str]
    route: Optional[str]                        # "tool" | "fallback" | "direct"
    final_answer: Optional[str]
    debug: Annotated[list[str], add]             # accumulates across nodes


# ─── Tools ───────────────────────────────────────────────────────────────────

def calculator_tool(expression: str) -> str:
    """Evaluate a math expression safely. Retries with common fixes on failure."""
    allowed = {
        "sqrt": math.sqrt, "pow": math.pow, "abs": abs,
        "round": round, "min": min, "max": max,
        "pi": math.pi, "e": math.e,
        "log": math.log, "log10": math.log10,
        "sin": math.sin, "cos": math.cos, "tan": math.tan,
    }

    # Build a list of expression variants to try
    variants = [expression.replace("^", "**")]

    # Auto-fix variants (tried only if the first eval fails)
    v0 = variants[0]
    fixes = [
        re.sub(r"\bmath\.", "", v0),                        # math.sqrt → sqrt
        v0.replace("×", "*").replace("÷", "/"),             # unicode operators
        re.sub(r"(\d+)\s*²", lambda m: f"pow({m.group(1)}, 2)", v0),  # 5²
        re.sub(r"(\d+)\s*³", lambda m: f"pow({m.group(1)}, 3)", v0),  # 5³
        re.sub(r"√\s*(\d+)",  lambda m: f"sqrt({m.group(1)})", v0),   # √4
    ]
    for f in fixes:
        if f != v0 and f not in variants:
            variants.append(f)

    last_error = None
    for variant in variants:
        try:
            result = eval(variant, {"__builtins__": {}}, allowed)
            return str(result)
        except Exception as e:
            last_error = e
            continue

    raise ValueError(f"Could not evaluate: {expression} → tried {variants}")


def search_tool(query: str) -> str:
    """Simulate a web search. Fails on queries containing '404' or 'error'."""
    if "404" in query.lower() or "error" in query.lower():
        raise ConnectionError("Search service returned 404 — resource not found")

    db = {
        "tunisia": (
            "Tunisia is a North African country bordering the Mediterranean Sea. "
            "Capital: Tunis · Population: ~12 million · Currency: Tunisian Dinar. "
            "Famous for the ancient ruins of Carthage, the Sahara Desert, "
            "and being the birthplace of the Arab Spring (2010)."
        ),
        "france": (
            "France is a Western European country. Capital: Paris · "
            "Population: ~67 million · Known for the Eiffel Tower, wine, and cheese."
        ),
        "python": (
            "Python is a high-level programming language created by "
            "Guido van Rossum in 1991. Emphasises readability and versatility."
        ),
        "ai": (
            "Artificial Intelligence simulates human intelligence in machines. "
            "Key areas: machine learning, NLP, computer vision, robotics."
        ),
        "langgraph": (
            "LangGraph is a framework for building stateful, multi-actor "
            "applications with LLMs, extending LangChain with graph-based workflows."
        ),
    }

    ql = query.lower().strip()
    for key, val in db.items():
        if key in ql:
            return val
    return f"Search for '{query}': No specific results in local database."


# ─── LLM (Ollama) ────────────────────────────────────────────────────────────

MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")

llm = ChatOllama(
    model=MODEL,
    temperature=0,
    num_predict=512,
)

logger.info(f"Using Ollama model: {MODEL}")


# ─── Ollama health check ─────────────────────────────────────────────────────

def check_ollama():
    """Verify Ollama is running and the model is available."""
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=3)
        models = [m["name"] for m in resp.json().get("models", [])]
        logger.info(f"Ollama running · models: {models}")
        if not any(MODEL in m for m in models):
            logger.warning(f"Model '{MODEL}' not found! Run: ollama pull {MODEL}")
    except Exception as e:
        logger.error(f"Cannot reach Ollama: {e} · Run: ollama serve")


check_ollama()


# ─── Safe LLM call wrapper ───────────────────────────────────────────────────

def safe_llm_invoke(messages, default="") -> str:
    """Call the LLM; return content string or default on failure."""
    try:
        response = llm.invoke(messages)
        content = response.content.strip()
        if not content:
            logger.warning("[LLM] Empty response")
            return default
        return content
    except Exception as e:
        logger.error(f"[LLM] Call failed: {e}")
        return default


# ─── Helper function for parsing search results ────────────────────────────────

def parse_search_results(query: str, search_results: str) -> str:
    """Use the LLM to parse search results and generate a good response."""
    prompt = (
        "You are given a user query and search results. Provide a helpful, natural answer.\n\n"
        "Rules:\n"
        "- Use the search results to answer the user's question\n"
        "- If search results are insufficient, provide a general answer based on your knowledge\n"
        "- Be concise but informative\n"
        "- Do not mention that you used search or that results were provided\n\n"
        f"User Query: {query}\n\n"
        f"Search Results: {search_results}\n\n"
        "Answer:"
    )

    answer = safe_llm_invoke(
        [HumanMessage(content=prompt)],
        default=search_results
    )
    return answer


# ─── Node: classify_input ────────────────────────────────────────────────────────────

def classify_input(state: AgentState) -> dict:
    query = state["query"]
    debug = [f"▸ Query: \"{query}\""]

    # LLM decides which tool to use and what input to provide
    prompt = (
        "You are an AI assistant that decides how to handle user queries.\n\n"
        "Available tools:\n"
        "1. calculator: For numeric calculations only (e.g., 12 * 8, sqrt(144) + 5, 2^10). "
        "NOT for equations with variables or solving equations.\n"
        "2. search: For factual information lookup (e.g., about countries, programming languages, concepts).\n"
        "3. direct: For everything else including equations with variables, conversational questions.\n\n"
        "Your task:\n"
        "- Decide which tool to use (calculator/search/direct)\n"
        "- If calculator: Provide the exact math expression in Python syntax. "
        "NO import statements, NO 'math.' prefix, NO semicolons. "
        "Use functions directly: sqrt(), pow(), sin(), cos(), tan(), log(), log10(), abs(). "
        "Use ** for exponentiation.\n"
        "- If search: Provide the search query (remove filler words like 'search for', 'tell me about').\n"
        "- If direct: No tool input needed.\n\n"
        "Format your response as JSON:\n"
        '{"tool": "calculator|search|direct", "input": "expression or query or null"}\n\n'
        f"Query: {query}\n\n"
        "Response:"
    )

    raw = safe_llm_invoke([HumanMessage(content=prompt)], default='{"tool": "direct", "input": null}')
    debug.append(f"▸ LLM decision: {raw}")

    # Parse the JSON response
    import json
    try:
        decision = json.loads(raw)
        tool_name = decision.get("tool", "direct")
        tool_input = decision.get("input")
    except json.JSONDecodeError:
        # Fallback if JSON parsing fails
        tool_name = "direct"
        tool_input = None
        debug.append("▸ Failed to parse LLM decision, defaulting to direct")

    debug.append(f"▸ Selected tool: {tool_name}")
    if tool_input:
        debug.append(f"▸ Tool input: \"{tool_input}\"")

    if tool_name == "direct":
        return {"tool_name": None, "tool_input": None, "route": "direct", "debug": debug}
    else:
        return {"tool_name": tool_name, "tool_input": tool_input, "route": None, "debug": debug}


# ─── Node: execute_tool ──────────────────────────────────────────────────────

def execute_tool(state: AgentState) -> dict:
    tool_name = state.get("tool_name")
    tool_input = state.get("tool_input")
    query = state["query"]
    debug = []

    debug.append(f"▸ Executing tool: {tool_name}")

    try:
        if tool_name == "calculator":
            # Use the expression prepared by LLM in classify_input
            if not tool_input:
                raise ValueError("LLM did not provide calculator expression")

            debug.append(f"▸ Using LLM-prepared expression: \"{tool_input}\"")
            debug.append(f"▸ Evaluating with calculator tool...")
            result = calculator_tool(tool_input)
            debug.append(f"▸ Calculator result: {result}")

        elif tool_name == "search":
            # Use the search query prepared by LLM in classify_input
            if not tool_input:
                raise ValueError("LLM did not provide search query")

            debug.append(f"▸ Using LLM-prepared search query: \"{tool_input}\"")
            debug.append(f"▸ Executing search...")
            search_results = search_tool(tool_input)
            debug.append(f"▸ Search results: {search_results[:100]}...")
            
            # Use LLM to parse search results and generate response
            debug.append("▸ Asking LLM to parse search results...")
            result = parse_search_results(query, search_results)
            debug.append(f"▸ Parsed answer: {result[:100]}...")

        else:
            raise ValueError(f"Unknown tool: {tool_name}")

        return {"tool_result": result, "tool_error": None, "route": "tool", "debug": debug}

    except Exception as e:
        debug.append(f"▸ Tool FAILED: {type(e).__name__}: {e}")
        return {"tool_error": str(e), "tool_result": None, "route": "fallback", "debug": debug}

# ─── Node: llm_answer ────────────────────────────────────────────────────────

def llm_answer(state: AgentState) -> dict:
    query = state["query"]
    route = state.get("route", "direct")
    debug = [f"▸ LLM answer mode: {route}"]

    system = "You are a helpful AI assistant. Answer directly and concisely."

    answer = safe_llm_invoke(
        [SystemMessage(content=system), HumanMessage(content=query)],
        default="I'm unable to generate a response right now.",
    )

    debug.append(f"▸ LLM response: \"{answer[:80]}...\"")
    return {"final_answer": answer, "route": route, "debug": debug}


# ─── Node: format_output ─────────────────────────────────────────────────────

def format_output(state: AgentState) -> dict:
    route = state.get("route", "direct")
    debug = [f"▸ Final route: {route}"]

    if route == "tool" and state.get("tool_result"):
        answer = state["tool_result"]
    elif state.get("final_answer"):
        answer = state["final_answer"]
    else:
        answer = "I couldn't process your request."

    debug.append(f"▸ Final answer: \"{answer[:80]}...\"")
    return {"final_answer": answer, "debug": debug}


# ─── Conditional edges ───────────────────────────────────────────────────────

def route_after_classify(state: AgentState) -> str:
    if state.get("tool_name"):
        return "execute_tool"
    return "llm_answer"


def route_after_tool(state: AgentState) -> str:
    if state.get("tool_error"):
        return "llm_answer"
    return "format_output"


# ─── Build the graph ────────────────────────────────────────────────────────

def build_agent():
    g = StateGraph(AgentState)

    g.add_node("classify_input", classify_input)
    g.add_node("execute_tool",   execute_tool)
    g.add_node("llm_answer",     llm_answer)
    g.add_node("format_output",  format_output)

    g.add_edge(START, "classify_input")
    g.add_conditional_edges("classify_input", route_after_classify)
    g.add_conditional_edges("execute_tool",   route_after_tool)
    g.add_edge("llm_answer",    "format_output")
    g.add_edge("format_output", END)

    return g.compile()


# ─── CLI test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    agent = build_agent()

    queries = [
        "What is sqrt(144) + 5?",
        "What is 12 * 8?",
        "Tell me something about Tunisia",
        "Search 404 test",
        "Who are you?",
        "What is 2^10?",
        "x² + 3x - 1 = 0 what's x?",
        "solve for x: 2x + 5 = 15",
    ]

    for q in queries:
        print(f"\n{'═'*60}")
        print(f"  Query: {q}")
        result = agent.invoke({"query": q, "debug": []})
        print(f"  Route:  {result.get('route')}")
        print(f"  Tool:   {result.get('tool_name')}")
        print(f"  Answer: {result.get('final_answer')}")
        print(f"  Debug:")
        for d in result.get("debug", []):
            print(f"    {d}")