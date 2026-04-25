import json                         # built-in Python library — used to parse JSON text into Python dictionaries
from openai import OpenAI           # OpenAI's Python library — OpenRouter uses the same format so we reuse this
from config import (                # import all model names and settings from config.py
    OPENROUTER_API_KEY,             # our secret API key
    OPENROUTER_BASE_URL,            # OpenRouter's URL
    AGENT1_MODEL,                   # "openai/gpt-4o-mini"
    AGENT2_MODEL,                   # "mistralai/mistral-7b-instruct"
    AGENT3_MODEL,                   # "google/gemini-2.5-flash"
    JUDGE_MODEL,                    # "anthropic/claude-sonnet-4-5"
)

# create ONE client object and reuse it for all API calls
# api_key  = your OpenRouter key from .env
# base_url = points to OpenRouter instead of OpenAI directly
client = OpenAI(
    api_key=OPENROUTER_API_KEY,     # your secret key
    base_url=OPENROUTER_BASE_URL,   # "https://openrouter.ai/api/v1"
)


# =============================================================================
# SYSTEM PROMPTS
# A system prompt tells the model WHO it is and HOW to respond
# We force JSON output so we can read position and confidence programmatically
# =============================================================================

# used when agent gives its FIRST response before any debate starts
AGENT_SYSTEM_PROMPT = """You are a news fact-checking agent in a multi-agent debate system.
Analyze whether the news claim is accurate based on the evidence provided.

You MUST respond ONLY with this exact JSON format — no extra text outside it:
{
  "position":   "SUPPORTED" or "REFUTED" or "UNCERTAIN",
  "confidence": a number between 0.0 and 1.0,
  "reasoning":  "explain your conclusion in 2-3 sentences",
  "evidence":   "quote the key evidence that led to your conclusion"
}"""

# used during debate rounds when agents argue against each other
DEBATE_SYSTEM_PROMPT = """You are a news fact-checking agent in an ongoing debate.
Review the other agents arguments and either defend or update your position.

You MUST respond ONLY with this exact JSON format — no extra text outside it:
{
  "position":      "SUPPORTED" or "REFUTED" or "UNCERTAIN",
  "confidence":    a number between 0.0 and 1.0,
  "reasoning":     "your updated reasoning after reviewing others arguments",
  "counter_point": "one specific thing you agree or disagree with from other agents"
}"""

# used by the judge who reads the full debate and gives the final verdict
JUDGE_SYSTEM_PROMPT = """You are the Judge in a multi-agent debate about a news claim.
Read all arguments carefully and give a final definitive verdict.

You MUST respond ONLY with this exact JSON format — no extra text outside it:
{
  "verdict":      "SUPPORTED" or "REFUTED" or "INSUFFICIENT_EVIDENCE",
  "confidence":   a number between 0.0 and 1.0,
  "reasoning":    "your full explanation for this verdict",
  "key_evidence": "the single most important piece of evidence that decided this"
}"""


# =============================================================================
# BASE FUNCTION — every agent uses this to talk to OpenRouter
# =============================================================================

def call_llm(model: str, system_prompt: str, user_prompt: str) -> dict:
    # model         = which LLM to use e.g. "openai/gpt-4o-mini"
    # system_prompt = tells the model who it is and how to behave
    # user_prompt   = the actual question or task for this specific call
    # returns       = a Python dict with position, confidence, reasoning, evidence

    try:                                                        # try to call the API — if it fails jump to except
        response = client.chat.completions.create(              # send the request to OpenRouter
            model=model,                                        # which model to use
            messages=[                                          # list of messages — same format as ChatGPT API
                {"role": "system", "content": system_prompt},  # system message sets the model's behaviour
                {"role": "user",   "content": user_prompt},    # user message is the actual task
            ],
            temperature=0.7,    # 0.0 = very consistent, 1.0 = very creative, 0.7 = balanced
            max_tokens=600,     # maximum length of response — 600 is enough for our JSON output
        )

        raw_text = response.choices[0].message.content.strip() # extract the text from the API response object
        return _parse_json(raw_text)                            # parse that text into a Python dict and return it

    except Exception as exc:                                    # if API call fails for any reason
        print(f"[ERROR] {model} failed: {exc}")                 # print the error so we can debug
        return _fallback(str(exc))                              # return a safe default so program doesn't crash


def _parse_json(text: str) -> dict:
    # models sometimes wrap their JSON in markdown like ```json ... ```
    # this function strips all that and returns a clean Python dict

    try:
        if "```json" in text:                                   # if model wrapped output in ```json
            text = text.split("```json")[1].split("```")[0].strip()  # extract just the JSON part

        elif "```" in text:                                     # if model wrapped output in ``` without json label
            text = text.split("```")[1].split("```")[0].strip() # extract just the content inside

        start = text.find("{")                                  # find the opening curly brace of JSON object
        end   = text.rfind("}") + 1                            # find the closing curly brace — rfind searches from right

        if start != -1 and end > start:                        # if we found both braces successfully
            return json.loads(text[start:end])                  # parse the JSON string into a Python dict

    except Exception:                                           # if parsing fails for any reason
        pass                                                    # fall through to fallback below

    return _fallback(text[:300])                                # if all parsing failed return safe default with raw text


def _fallback(reason: str) -> dict:
    # returns a safe default dict when a model fails or response cant be parsed
    # prevents the whole debate from crashing because one agent had a bad response
    return {
        "position":   "UNCERTAIN",    # default to uncertain — safest assumption when we dont know
        "confidence": 0.5,            # 50% confidence — middle of the road
        "reasoning":  reason,         # include the error or raw text so we can debug
        "evidence":   "unavailable",  # no evidence since the call failed
    }


# =============================================================================
# AGENT FUNCTIONS — each agent has its own function locked to its own model
# =============================================================================

def agent1_call(user_prompt: str) -> dict:
    # Agent 1 — GPT-4o-mini — called for FIRST response before any debate
    return call_llm(AGENT1_MODEL, AGENT_SYSTEM_PROMPT, user_prompt)

def agent2_call(user_prompt: str) -> dict:
    # Agent 2 — Mistral 7B — called for FIRST response before any debate
    return call_llm(AGENT2_MODEL, AGENT_SYSTEM_PROMPT, user_prompt)

def agent3_call(user_prompt: str) -> dict:
    # Agent 3 — Gemini 2.5 Flash — called for FIRST response before any debate
    return call_llm(AGENT3_MODEL, AGENT_SYSTEM_PROMPT, user_prompt)


def agent1_debate(user_prompt: str) -> dict:
    # Agent 1 — GPT-4o-mini — called DURING debate rounds to argue back
    return call_llm(AGENT1_MODEL, DEBATE_SYSTEM_PROMPT, user_prompt)

def agent2_debate(user_prompt: str) -> dict:
    # Agent 2 — Mistral 7B — called DURING debate rounds to argue back
    return call_llm(AGENT2_MODEL, DEBATE_SYSTEM_PROMPT, user_prompt)

def agent3_debate(user_prompt: str) -> dict:
    # Agent 3 — Gemini 2.5 Flash — called DURING debate rounds to argue back
    return call_llm(AGENT3_MODEL, DEBATE_SYSTEM_PROMPT, user_prompt)


def judge_call(user_prompt: str) -> dict:
    # Judge — Claude Sonnet — reads full debate and gives final verdict
    return call_llm(JUDGE_MODEL, JUDGE_SYSTEM_PROMPT, user_prompt)