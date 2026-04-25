import os                                          # built-in Python library — lets us read from environment variables
from dotenv import load_dotenv                     # reads your .env file and loads the API key into the program

load_dotenv()                                      # run this function — now OPENROUTER_API_KEY is available

# ── OpenRouter API settings ───────────────────────────────────────────────────

OPENROUTER_API_KEY  = os.getenv("OPENROUTER_API_KEY", "")   # read your secret key from .env file
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"         # OpenRouter API address — all requests go here

# ── Model assignment — one different model per agent ─────────────────────────

AGENT1_MODEL = "openai/gpt-4o-mini"               # Agent 1 — GPT-4o-mini — best at reading web search results
AGENT2_MODEL = "x-ai/grok-3-mini-beta"    # Agent 2 — Grok-3-mini-beta — good at structured factual reading
AGENT3_MODEL = "google/gemini-2.5-flash"          # Agent 3 — Gemini 2.5 Flash — strong logical reasoning
JUDGE_MODEL  = "anthropic/claude-sonnet-4-5"      # Judge — Claude Sonnet — strongest model, different family from all agents

# ── Debate control settings ───────────────────────────────────────────────────

MAX_DEBATE_ROUNDS    = 2     # agents argue back and forth maximum 2 times before judge is forced to decide
CONFIDENCE_THRESHOLD = 0.80  # if ALL 3 agents agree AND score above 80% confidence — skip debate entirely