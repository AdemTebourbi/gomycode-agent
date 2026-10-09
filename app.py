import os
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
from agent import build_agent, check_ollama

load_dotenv()

app = Flask(__name__)

# Build agent at startup so we see Ollama check immediately
agent = build_agent()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    query = data.get("message", "").strip()

    if not query:
        return jsonify({"error": "Empty message"}), 400

    try:
        result = agent.invoke({"query": query, "debug": []})
        return jsonify({
            "answer": result.get("final_answer", "No answer generated."),
            "route":  result.get("route", "unknown"),
            "tool":   result.get("tool_name"),
            "debug":  result.get("debug", []),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    """Check if Ollama is reachable."""
    import requests as req
    try:
        resp = req.get("http://localhost:11434/api/tags", timeout=3)
        models = [m["name"] for m in resp.json().get("models", [])]
        return jsonify({"status": "ok", "models": models})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 503


if __name__ == "__main__":
    app.run(debug=True, port=5000)