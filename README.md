# 🔍 Multi-Agent News Fact Verification System

A production-grade fact verification system powered by Multi-Agent Debate (MAD) using four heterogeneous LLMs that independently research, debate, and verify news claims in real time.

---

## 🧠 How It Works

```mermaid
flowchart TD

    A[News Claim Input]

    A --> B[Independent Research]

    B --> C1["Agent 1<br/>GPT-4o-mini<br/>Live Web Search"]
    B --> C2["Agent 2<br/>Grok 4 Fast<br/>Wikipedia Retrieval"]
    B --> C3["Agent 3<br/>Gemini 2.5<br/>Pure Reasoning"]

    C1 --> D[Confidence Gate<br/>DOWN Framework - Eo et al.]
    C2 --> D
    C3 --> D

    D -->|All Agree & Confidence > 80%| G[Judge<br/>Claude Sonnet]

    D -->|Disagreement or Low Confidence| E[Debate Round 1]
    E --> F[Debate Round 2]
    F --> G

    G --> H{Final Verdict}

    H --> I[✅ SUPPORTED]
    H --> J[❌ REFUTED]
    H --> K[⚠️ INSUFFICIENT EVIDENCE]
```

---

## 🤖 Agent Design

| Agent | Model | Tool | Role |
|-------|-------|------|------|
| Agent 1 | `openai/gpt-4o-mini` | DuckDuckGo Web Search | Finds latest web evidence |
| Agent 2 | `x-ai/grok-4-fast` | Wikipedia API | Provides background context |
| Agent 3 | `google/gemini-2.5-flash` | None | Pure logical reasoning |
| Judge | `anthropic/claude-sonnet-4-5` | Full debate transcript | Final verdict |

> All four agents use different model families (OpenAI, xAI, Google, Anthropic) — a deliberate design choice based on Zhou et al. A-HMAD which shows heterogeneous agents produce higher quality debates than identical ones. The judge being from a different family than all debaters reduces adjudication bias as warned by Liang et al.

---

## 📂 Project Structure

```text
mad-fact-verifier/
│
├── config.py
│   └── Centralized model configuration and system settings
│
├── tools.py
│   └── Web search and Wikipedia retrieval utilities
│
├── llm_caller.py
│   └── OpenRouter API integration for all agents
│
├── debate_graph.py
│   └── Multi-agent debate workflow (LangGraph)
│
├── load_fever.py
│   └── Downloads claims from the Climate-FEVER benchmark
│
├── evaluate.py
│   └── Evaluation pipeline and accuracy metrics
│
├── main.py
│   └── Interactive CLI for fact-checking claims
│
├── claims.json
│   └── Benchmark claims dataset
│
├── evaluation_results.json
│   └── Evaluation results used in experiments
│
├── .env.example
│   └── Environment variable template
│
└── README.md
    └── Project documentation
```

---

## ⚙️ Setup

### 1. Clone the repository
```bash
git clone https://github.com/farisansari10/mad-fact-verifier.git
cd mad-fact-verifier
```

### 2. Install dependencies
```bash
pip3 install langgraph openai wikipedia ddgs python-dotenv datasets
```

### 3. Create your `.env` file
```bash
cp .env.example .env
```
Open `.env` and add your OpenRouter API key:
OPENROUTER_API_KEY=your-key-here
Get your key at [openrouter.ai](https://openrouter.ai)

### 4. Generate test claims
```bash
python3 load_fever.py
```

---

## 🚀 Usage

### Interactive mode — verify any claim
```bash
python3 main.py
```

### Evaluation mode — benchmark accuracy
```bash
python3 evaluate.py
```
EVALUATION RESULTS
Accuracy          : 53.3%  (8/15)
Avg Confidence    : 0.89
Avg Debate Rounds : 1.0
Total Time        : 4912s

---

## 📊 Evaluation

Evaluated on 15 claims from the [climate_fever](https://huggingface.co/datasets/climate_fever) benchmark dataset — a peer-reviewed dataset of climate science claims with labels SUPPORTED, REFUTED, and INSUFFICIENT_EVIDENCE.

| Metric | Value |
|--------|-------|
| Accuracy | 53.3% (8/15) |
| Avg Confidence | 0.89 |
| Avg Debate Rounds | 1.0 |
| Total Evaluation Time | ~82 minutes |

> Note: climate_fever is a challenging benchmark with nuanced scientific claims. Manual inspection revealed several cases where our system's prediction was arguably more accurate than the dataset's ground truth label — consistent with findings from Xie et al. who showed even GPT-4 struggles on real-world fact verification tasks.

---

## 📚 Research Foundation

This system directly implements concepts from the following papers reviewed in the companion survey (Assignment 2):

| Paper | Concept Implemented |
|-------|---------------------|
| Du et al. (2024) | Foundational multi-agent debate protocol |
| Zhou et al. (2025) A-HMAD | Heterogeneous agents with different models and tools |
| Eo et al. (2025) DOWN | Confidence gate — debate only when necessary |
| Liang et al. (2024) | Judge from different model family to reduce bias |
| Jeong et al. (2026) Tool-MAD | Agents equipped with different external retrieval tools |

---

## 🔧 Configuration

All settings live in `config.py`. To swap a model change it there only — nothing else needs updating.

```python
AGENT1_MODEL = "openai/gpt-4o-mini"
AGENT2_MODEL = "x-ai/grok-4-fast"
AGENT3_MODEL = "google/gemini-2.5-flash"
JUDGE_MODEL  = "anthropic/claude-sonnet-4-5"

MAX_DEBATE_ROUNDS    = 2
CONFIDENCE_THRESHOLD = 0.80
```

---

## ⚠️ Limitations

- No persistent memory between sessions — each claim starts fresh
- Dependent on web search quality — misleading web results affect accuracy
- Climate_fever dataset contains noisy labels on nuanced scientific claims
- Debate rounds increase latency — average 5-6 minutes per claim on evaluation

---

## 🔮 Future Work

- Add persistent memory to avoid re-debating previously verified claims
- Integrate News API and Google Search API for higher quality evidence
- Build a FastAPI backend and web frontend for public access
- Expand evaluation to larger benchmarks like FEVER or LIAR dataset

---

## 👤 Author

**Faris Ansari**
FAST-NUCES, MS-CS
faris.ansari10@gmail.com

---

## 📄 License

MIT License — free to use, modify, and distribute.