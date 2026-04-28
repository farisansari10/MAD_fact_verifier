import json
from openai import OpenAI
from config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    AGENT1_MODEL,
    AGENT2_MODEL,
    AGENT3_MODEL,
    JUDGE_MODEL,
)

client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url=OPENROUTER_BASE_URL,
)


AGENT_SYSTEM_PROMPT = """You are a news fact-checking agent in a multi-agent debate system.
Analyze whether the news claim is accurate based on the evidence provided.

You MUST respond ONLY with this exact JSON format — no extra text outside it:
{
  "position":   "SUPPORTED" or "REFUTED" or "UNCERTAIN",
  "confidence": a number between 0.0 and 1.0,
  "reasoning":  "explain your conclusion in 2-3 sentences",
  "evidence":   "quote the key evidence that led to your conclusion"
}"""

DEBATE_SYSTEM_PROMPT = """You are a news fact-checking agent in an ongoing debate.
Review the other agents arguments and either defend or update your position.

You MUST respond ONLY with this exact JSON format — no extra text outside it:
{
  "position":      "SUPPORTED" or "REFUTED" or "UNCERTAIN",
  "confidence":    a number between 0.0 and 1.0,
  "reasoning":     "your updated reasoning after reviewing others arguments",
  "counter_point": "one specific thing you agree or disagree with from other agents"
}"""

JUDGE_SYSTEM_PROMPT = """You are the Judge in a multi-agent debate about a news claim.
Read all arguments carefully and give a final definitive verdict.

You MUST respond ONLY with this exact JSON format — no extra text outside it:
{
  "verdict":      "SUPPORTED" or "REFUTED" or "INSUFFICIENT_EVIDENCE",
  "confidence":   a number between 0.0 and 1.0,
  "reasoning":    "your full explanation for this verdict",
  "key_evidence": "the single most important piece of evidence that decided this"
}"""


def call_llm(model: str, system_prompt: str, user_prompt: str) -> dict:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.7,
            max_tokens=600,
        )

        raw_text = response.choices[0].message.content.strip()
        return _parse_json(raw_text)

    except Exception as exc:
        print(f"[ERROR] {model} failed: {exc}")
        return _fallback(str(exc))


def _parse_json(text: str) -> dict:
    try:
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()

        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        start = text.find("{")
        end   = text.rfind("}") + 1

        if start != -1 and end > start:
            return json.loads(text[start:end])

    except Exception:
        pass

    return _fallback(text[:300])


def _fallback(reason: str) -> dict:
    return {
        "position":   "UNCERTAIN",
        "confidence": 0.5,
        "reasoning":  reason,
        "evidence":   "unavailable",
    }


def agent1_call(user_prompt: str) -> dict:
    return call_llm(AGENT1_MODEL, AGENT_SYSTEM_PROMPT, user_prompt)

def agent2_call(user_prompt: str) -> dict:
    return call_llm(AGENT2_MODEL, AGENT_SYSTEM_PROMPT, user_prompt)

def agent3_call(user_prompt: str) -> dict:
    return call_llm(AGENT3_MODEL, AGENT_SYSTEM_PROMPT, user_prompt)


def agent1_debate(user_prompt: str) -> dict:
    return call_llm(AGENT1_MODEL, DEBATE_SYSTEM_PROMPT, user_prompt)

def agent2_debate(user_prompt: str) -> dict:
    return call_llm(AGENT2_MODEL, DEBATE_SYSTEM_PROMPT, user_prompt)

def agent3_debate(user_prompt: str) -> dict:
    return call_llm(AGENT3_MODEL, DEBATE_SYSTEM_PROMPT, user_prompt)


def judge_call(user_prompt: str) -> dict:
    return call_llm(JUDGE_MODEL, JUDGE_SYSTEM_PROMPT, user_prompt)