# ============================================================
# SCHOLARAI - LLM SERVICE
# File: llm.py
#
# Primary connection to Ollama (local LLM), with automatic fallback
# to Gemini API when running in environments where Ollama is offline.
# ============================================================

import os
import requests
from dotenv import load_dotenv

load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2:latest")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

AVAILABLE_MODELS = [
    "llama3.2:latest",
    "llama3.2:1b",
    "qwen2.5:3b",
    "mistral:latest",
    "phi3:latest",
    "gemini-1.5-flash",
]

def set_llm_model(model_name: str) -> str:
    global LLM_MODEL
    cleaned = model_name.strip()
    if cleaned:
        LLM_MODEL = cleaned
    return LLM_MODEL

REQUEST_TIMEOUT = 120

SYSTEM_PROMPT = """You are ScholarAI, an intelligent Student Assistant.

Your core directive:
Answer the user's CURRENT question accurately and clearly.
Do not repeat previous answers unless the user explicitly asks for them.
Use provided document context only when it is relevant.
If the document does not contain the answer, answer using general knowledge when appropriate.
Never assume that the current question is the same as the previous question.

RESPONSE INSTRUCTIONS:
- Directly answer the current question.
- Give a clear explanation.
- Use simple student-friendly language.
- Use bullet points when useful.
- Give examples when useful.
- Do not give unrelated information.
- Do not copy the previous answer.
- If the question is unclear, ask a short clarification question."""


class LLMError(Exception):
    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(message)


def _generate_offline_response(prompt_or_msg: str, tool_name: str = "Chat") -> str:
    """Provides a clear error message when all AI services are unavailable."""
    return (
        f"**Error**: The AI model service is currently unavailable or misconfigured.\n\n"
        f"I am unable to process your request about '{prompt_or_msg[:50]}...' because both the local Ollama instance and the Gemini API fallback failed to respond. Please check the backend console logs and ensure `GEMINI_API_KEY` is configured correctly."
    )


def _call_gemini_fallback(messages: list[dict], temperature: float = 0.7) -> str:
    """Fallback to gemini-3.6-flash if Ollama is not running."""
    if not GEMINI_API_KEY:
        return _generate_offline_response(messages[-1].get("content", "Study Inquiry") if messages else "Study Inquiry")

    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash:generateContent?key={GEMINI_API_KEY}"
        
        contents = []
        for m in messages:
            if m.get("role") == "system":
                continue
            role = "user" if m.get("role") == "user" else "model"
            
            # Gemini requires alternating roles, combine if same as previous
            if contents and contents[-1]["role"] == role:
                contents[-1]["parts"][0]["text"] += "\n" + m.get("content", "")
            else:
                contents.append({
                    "role": role,
                    "parts": [{"text": m.get("content", "")}]
                })
        
        # Ensure the last message is from the user for generateContent
        if contents and contents[-1]["role"] == "model":
            contents.append({
                "role": "user",
                "parts": [{"text": "Continue"}]
            })
        
        if not contents:
            return "No valid request to answer."

        payload = {
            "contents": contents,
            "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "generationConfig": {
                "temperature": temperature,
                "topK": 40,
                "topP": 0.9
            }
        }

        resp = requests.post(url, json=payload, timeout=45)
        resp.raise_for_status()
        data = resp.json()
        candidates = data.get("candidates", [])
        if candidates and "content" in candidates[0]:
            parts = candidates[0]["content"].get("parts", [])
            if parts:
                return parts[0].get("text", "")
        return "No response generated."
    except Exception:
        last_msg = messages[-1].get("content", "Study Inquiry") if messages else "Study Inquiry"
        return _generate_offline_response(last_msg)


def _call_ollama(messages: list[dict], temperature: float = 0.7, num_predict: int = 2500) -> str:
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": LLM_MODEL,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": num_predict,
                },
            },
            timeout=60.0,
        )
        response.raise_for_status()
        result = response.json()
        return result["message"]["content"]
    except Exception:
        # Fall back to Gemini or structured offline response
        return _call_gemini_fallback(messages, temperature=temperature)




# ------------------------------------------------------------
# BASIC CHAT
# ------------------------------------------------------------

def generate_response(message: str, history: list[dict] | None = None) -> str:
    if history is None:
        history = []

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": message})

    return _call_ollama(messages)


def generate_chat_response(conversation: list[dict]) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(conversation)
    return _call_ollama(messages)


# ------------------------------------------------------------
# RAG RESPONSE
# ------------------------------------------------------------

def generate_rag_response(question: str, context: str) -> str:
    prompt = f"""Answer the student's question using the provided study material. Employ creative thinking and step-by-step reasoning to train the student on the underlying process.

IMPORTANT:
- Use the study material as the main source, but synthesize it creatively to explain the process clearly.
- Walk through the problem step-by-step (Chain of Thought) before giving the final answer.
- Do not invent unsupported information.
- If the answer is not present, say that it is not available in the uploaded study material.
- Explain clearly, and use real-world analogies or creative examples to aid understanding.

STUDY MATERIAL
================
{context}
================

STUDENT QUESTION
================
{question}
================

REASONING AND ANSWER:
"""
    return _call_ollama([{"role": "user", "content": prompt}], temperature=0.7)


# ------------------------------------------------------------
# AI STUDY TOOLS
# ------------------------------------------------------------

def summarize_note(content: str) -> str:
    prompt = f"""Conduct a thorough, high-retention academic synthesis of the following study material.

Structure requirements:
1. Executive Abstract & Core Thesis
2. Detailed Key Concepts & Technical Principles (with structured bullets)
3. Formal Definitions & Nuances
4. Practical Implications & Applications
5. Critical Review Takeaways

STUDY NOTE:
{content}
"""
    return generate_response(prompt)


def explain_concept(content: str) -> str:
    prompt = f"""Provide an in-depth, rigorous, and deeply explained academic guide for the following concept.

Structure requirements:
1. Formal Definition & Historical/Theoretical Origins
2. First-Principles Mechanical Breakdown (how it works internally)
3. Concrete Practical Example with Code or Step-by-Step Logic
4. Mathematical or Algorithmic Formulations (if applicable)
5. Comparative Trade-Offs, Pros & Cons
6. Real-World Industry & Academic Applications
7. Conceptual Edge Cases & Common Pitfalls

CONCEPT:
{content}
"""
    return generate_response(prompt)


def generate_questions(content: str) -> str:
    prompt = f"""Generate important exam questions from the following study material.

Create:
- 5 short-answer questions
- 5 long-answer questions
- 5 MCQs with answers

STUDY MATERIAL:
{content}
"""
    return generate_response(prompt)


def generate_flashcards(content: str) -> str:
    prompt = f"""Create study flashcards from the following material.

Format:
Q1: Question
A1: Answer

Create 10 useful flashcards.

STUDY MATERIAL:
{content}
"""
    return generate_response(prompt)


def generate_model_training_guide(content: str) -> str:
    prompt = f"""Act as a senior AI/ML research engineer and educator. Provide a comprehensive, rigorous academic and practical guide on how to train, fine-tune, and evaluate a neural/LLM model (such as {LLM_MODEL}) based on or related to the following material.

Structure your response with:
1. **Model Architecture & Objective**: Pre-training vs. Supervised Fine-Tuning (SFT) vs. Preference Alignment (DPO/RLHF).
2. **Dataset Curation & Formatting**: How to structure training data (e.g., ShareGPT or Alpaca JSONL schemas), tokenization, and context window padding.
3. **Training & Fine-Tuning Strategy (LoRA / QLoRA)**:
   - Rank $r$, Alpha $\\alpha$, target projection modules (q_proj, v_proj, k_proj, o_proj)
   - 4-bit NF4 Quantization and memory footprint
4. **Hyperparameters & Optimization**:
   - Learning rate (e.g. 2e-4 with Cosine Annealing)
   - Batch size & Gradient Accumulation
   - Optimizer (PagedAdamW 8-bit) and Weight Decay
5. **Complete Executable Training Script**: (Provide a clean PyTorch / HuggingFace `trl.SFTTrainer` or Unsloth script)
6. **Loss Curves & Evaluation Benchmarks**: Perplexity, MMLU/GSM8K evaluation, and overfitting prevention.

STUDY MATERIAL / TOPIC:
{content}
"""
    return generate_response(prompt)


def deep_research_synthesis(content: str) -> str:
    prompt = f"""Conduct a deep academic research analysis and comprehensive synthesis on the following material.

Include:
- Core Theoretical Foundations
- Mathematical Formulation & Derivations (if applicable)
- Algorithmic Mechanics & Edge Cases
- Empirical Performance & Benchmark Insights
- Academic Summary & Self-Testing Review Questions

MATERIAL:
{content}
"""
    return generate_response(prompt)


def perform_deep_search(query: str, context: str = "") -> dict:
    """Executes a multi-phase deep research search engine query."""
    source_context = f"\nRELEVANT SOURCE CONTEXT:\n{context}\n" if context else ""
    prompt = f"""You are the ScholarAI Deep Search & Model Engineering Engine powered by {LLM_MODEL}.

Perform an exhaustive, multi-step deep academic search and structural analysis for the following inquiry:
QUERY: {query}
{source_context}

Execute the following deep search phases:
### Phase 1: Query Decomposition & Core Scientific Principles
Break the topic down into its fundamental theoretical components and conceptual prerequisites.

### Phase 2: Technical Architecture & Training / Algorithmic Mechanics
Detail how the underlying model, engine, or algorithm operates, including mathematical definitions and internal representations.

### Phase 3: Model Training & Implementation Blueprint
Provide concrete engineering parameters, hyperparameters, dataset formats, or code implementation for practical execution.

### Phase 4: Critical Comparative Analysis & Citations
Synthesize trade-offs, empirical benchmarks, and recommended academic references.
"""
    deep_answer = generate_response(prompt)
    return {
        "engine": LLM_MODEL,
        "mode": "deep_search",
        "answer": deep_answer,
        "phases": [
            "1. Query Decomposition & Core Principles",
            "2. Technical Architecture & Algorithmic Mechanics",
            "3. Model Training & Implementation Blueprint",
            "4. Critical Analysis & Academic Citations"
        ]
    }


AI_TOOL_FUNCTIONS = {
    "Summarize Note": summarize_note,
    "Explain Concept": explain_concept,
    "Generate Questions": generate_questions,
    "Generate Flashcards": generate_flashcards,
    "Train & Fine-Tune Guide": generate_model_training_guide,
    "Deep Research & Synthesis": deep_research_synthesis,
}


def run_ai_tool(tool: str, content: str) -> str:
    func = AI_TOOL_FUNCTIONS.get(tool)
    if not func:
        raise LLMError(f"Unknown AI tool '{tool}'", status_code=400)
    return func(content)


def get_model_info() -> dict:
    return {
        "provider": "Ollama / Gemini",
        "model": LLM_MODEL,
        "endpoint": OLLAMA_URL,
        "available_models": AVAILABLE_MODELS,
        "deep_search_enabled": True,
        "status": "ready",
    }
