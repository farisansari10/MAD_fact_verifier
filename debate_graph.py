import json                                        # built-in Python library — used to handle JSON data
from typing import TypedDict                       # lets us define exactly what keys a dictionary must have
from langgraph.graph import StateGraph, START, END # StateGraph = flowchart builder, START and END = entry and exit points
from config import MAX_DEBATE_ROUNDS, CONFIDENCE_THRESHOLD  # debate settings from config.py
from tools import search_web, search_wikipedia     # the two search functions from tools.py
from llm_caller import (                           # import all agent functions from llm_caller.py
    agent1_call, agent2_call, agent3_call,         # first response functions — called before debate starts
    agent1_debate, agent2_debate, agent3_debate,   # rebuttal functions — called during debate rounds
    judge_call,                                    # judge function — called at the end for final verdict
)


# =============================================================================
# STATE
# This is a shared notebook that every node reads from and writes to
# LangGraph automatically passes this between every node in the graph
# Think of it like a baton passed between runners in a relay race
# =============================================================================

class DebateState(TypedDict):         # TypedDict = every key has a fixed type — no surprises
    claim:            str             # the news claim being verified e.g. "Pakistan won the Champions Trophy"
    round_number:     int             # tracks how many debate rounds have happened so far — starts at 0
    agent1_response:  dict            # Agent 1's latest response — has keys: position, confidence, reasoning, evidence
    agent2_response:  dict            # Agent 2's latest response
    agent3_response:  dict            # Agent 3's latest response
    debate_history:   str             # full running log of everything said — judge reads this at the end
    final_verdict:    str             # SUPPORTED / REFUTED / INSUFFICIENT_EVIDENCE — filled in by judge
    final_reasoning:  str             # judge's explanation for the verdict
    final_confidence: float           # judge's confidence score between 0.0 and 1.0


# =============================================================================
# NODE 1 — Agent 1: GPT-4o-mini + Web Search
# A node is just a function that receives the current state and returns updates
# =============================================================================

def node_agent1(state: DebateState) -> dict:       # takes current state, returns dict of updates
    print("\n  [Agent 1 - GPT-4o-mini] Searching the web...")

    web_results = search_web(state["claim"])        # search DuckDuckGo using the claim as the query

    # build the full message Agent 1 will receive
    # f-string = a string where {variables} are replaced with their actual values
    prompt = f"""News claim to verify: "{state['claim']}"

Web search results:
{web_results}

Based on this evidence, is the claim SUPPORTED, REFUTED, or UNCERTAIN?"""

    response = agent1_call(prompt)                  # send prompt to GPT-4o-mini via OpenRouter — get back a dict

    history = state.get("debate_history", "")       # get current debate history — empty string if nothing yet
    history += f"\n[ROUND 0 - Agent 1 (GPT-4o-mini)]\n"          # add header for this entry
    history += f"Position  : {response.get('position')}\n"        # log its position
    history += f"Confidence: {response.get('confidence')}\n"      # log its confidence score
    history += f"Reasoning : {response.get('reasoning')}\n"       # log its reasoning
    history += f"Evidence  : {response.get('evidence')}\n"        # log its evidence

    return {                                        # return ONLY the keys we are updating — LangGraph merges this into state
        "agent1_response": response,               # save Agent 1's full response into state
        "debate_history":  history,                # save updated debate log into state
    }


# =============================================================================
# NODE 2 — Agent 2: Mistral 7B + Wikipedia
# =============================================================================

def node_agent2(state: DebateState) -> dict:       # takes current state, returns dict of updates
    print("  [Agent 2 - Grok 3 Mini Beta] Checking Wikipedia...")

    wiki_results = search_wikipedia(state["claim"]) # search Wikipedia using the claim as the query

    prompt = f"""News claim to verify: "{state['claim']}"

Wikipedia background:
{wiki_results}

Based on this background information, is the claim SUPPORTED, REFUTED, or UNCERTAIN?"""

    response = agent2_call(prompt)                  # send prompt to Mistral 7B via OpenRouter

    history = state.get("debate_history", "")       # get existing debate history from state
    history += f"\n[ROUND 0 - Agent 2 (Grok 3 Mini Beta)]\n"
    history += f"Position  : {response.get('position')}\n"
    history += f"Confidence: {response.get('confidence')}\n"
    history += f"Reasoning : {response.get('reasoning')}\n"
    history += f"Evidence  : {response.get('evidence')}\n"

    return {
        "agent2_response": response,               # save Agent 2's response into state
        "debate_history":  history,                # save updated debate log into state
    }


# =============================================================================
# NODE 3 — Agent 3: Gemini 2.5 Flash + Pure Reasoning (no tools)
# =============================================================================

def node_agent3(state: DebateState) -> dict:       # takes current state, returns dict of updates
    print("  [Agent 3 - Gemini 2.5 Flash] Reasoning independently...")

    # Agent 3 deliberately gets NO search results
    # this forces it to reason differently from Agent 1 and Agent 2
    # diversity of reasoning = better debate quality — based on Zhou et al. A-HMAD paper
    prompt = f"""News claim to verify: "{state['claim']}"

Use only your own knowledge and logical reasoning — no external tools available.
Is this claim SUPPORTED, REFUTED, or UNCERTAIN?"""

    response = agent3_call(prompt)                  # send prompt to Gemini 2.5 Flash via OpenRouter

    history = state.get("debate_history", "")       # get existing debate history from state
    history += f"\n[ROUND 0 - Agent 3 (Gemini 2.5 Flash)]\n"
    history += f"Position  : {response.get('position')}\n"
    history += f"Confidence: {response.get('confidence')}\n"
    history += f"Reasoning : {response.get('reasoning')}\n"
    history += f"Evidence  : {response.get('evidence')}\n"

    return {
        "agent3_response": response,               # save Agent 3's response into state
        "debate_history":  history,                # save updated debate log into state
    }


# =============================================================================
# CONFIDENCE GATE — do we skip debate or start arguing?
# This is a conditional edge function — it returns a string
# LangGraph uses that string to decide which node to go to next
# Implements the DOWN paper: "Debate Only When Necessary" by Eo et al.
# =============================================================================

def confidence_gate(state: DebateState) -> str:    # returns "skip" or "debate"

    r1 = state.get("agent1_response", {})          # get Agent 1's response from state
    r2 = state.get("agent2_response", {})          # get Agent 2's response from state
    r3 = state.get("agent3_response", {})          # get Agent 3's response from state

    pos1 = r1.get("position", "UNCERTAIN")         # get Agent 1's position — default UNCERTAIN if missing
    pos2 = r2.get("position", "UNCERTAIN")         # get Agent 2's position
    pos3 = r3.get("position", "UNCERTAIN")         # get Agent 3's position

    conf1 = float(r1.get("confidence", 0.5))       # get Agent 1's confidence as float — default 0.5
    conf2 = float(r2.get("confidence", 0.5))       # get Agent 2's confidence as float
    conf3 = float(r3.get("confidence", 0.5))       # get Agent 3's confidence as float

    all_agree     = (pos1 == pos2 == pos3) and pos1 != "UNCERTAIN"  # True if all three have same non-uncertain position
    all_high_conf = (conf1 >= CONFIDENCE_THRESHOLD and              # True if all three are above 0.80 confidence
                     conf2 >= CONFIDENCE_THRESHOLD and
                     conf3 >= CONFIDENCE_THRESHOLD)

    if all_agree and all_high_conf:                # if everyone agrees AND everyone is very confident
        print("  [Gate] All agents agree with high confidence — skipping debate.")
        return "skip"                              # tell LangGraph to go directly to judge node

    print("  [Gate] Disagreement or low confidence detected — starting debate.")
    return "debate"                                # tell LangGraph to go to debate_round node


# =============================================================================
# NODE 4 — Debate Round: agents read each other's arguments and respond
# =============================================================================

def node_debate_round(state: DebateState) -> dict:
    round_num = state.get("round_number", 0) + 1  # increment round counter — first call makes it 1
    print(f"\n  [Debate Round {round_num}] Agents reviewing each other's arguments...")

    history = state.get("debate_history", "")      # get full debate history so far

    # ── Agent 1 rebuttal ──────────────────────────────────────────────────────
    fresh_web = search_web(state["claim"] + " latest news")  # search again with "latest news" for fresher results

    prompt1 = f"""Claim: "{state['claim']}"

Full debate so far:
{history}

Fresh web search:
{fresh_web}

You are Agent 1. Read the other agents arguments carefully and update your position."""

    r1 = agent1_debate(prompt1)                    # send to GPT-4o-mini for its rebuttal response
    history += f"\n[ROUND {round_num} - Agent 1 rebuttal]\n"
    history += f"Position  : {r1.get('position')}\n"
    history += f"Confidence: {r1.get('confidence')}\n"
    history += f"Counter   : {r1.get('counter_point')}\n"  # what it specifically disagrees with

    # ── Agent 2 rebuttal ──────────────────────────────────────────────────────
    prompt2 = f"""Claim: "{state['claim']}"

Full debate so far:
{history}

You are Agent 2. Read the other agents arguments carefully and update your position."""

    r2 = agent2_debate(prompt2)                    # send to Grok 3 Mini Beta for its rebuttal response
    history += f"\n[ROUND {round_num} - Agent 2 rebuttal]\n"
    history += f"Position  : {r2.get('position')}\n"
    history += f"Confidence: {r2.get('confidence')}\n"
    history += f"Counter   : {r2.get('counter_point')}\n"

    # ── Agent 3 rebuttal ──────────────────────────────────────────────────────
    prompt3 = f"""Claim: "{state['claim']}"

Full debate so far:
{history}

You are Agent 3. Read the other agents arguments carefully and update your reasoning."""

    r3 = agent3_debate(prompt3)                    # send to Gemini 2.5 Flash for its rebuttal response
    history += f"\n[ROUND {round_num} - Agent 3 rebuttal]\n"
    history += f"Position  : {r3.get('position')}\n"
    history += f"Confidence: {r3.get('confidence')}\n"
    history += f"Counter   : {r3.get('counter_point')}\n"

    return {                                        # return all updates to state
        "round_number":    round_num,              # updated round counter
        "agent1_response": r1,                     # Agent 1's updated response
        "agent2_response": r2,                     # Agent 2's updated response
        "agent3_response": r3,                     # Agent 3's updated response
        "debate_history":  history,                # full updated debate log
    }


# =============================================================================
# SHOULD WE KEEP DEBATING or go to judge?
# Another conditional edge — returns "continue" or "judge"
# =============================================================================

def should_continue(state: DebateState) -> str:
    if state.get("round_number", 0) >= MAX_DEBATE_ROUNDS:  # if we hit max rounds limit (2)
        print("  [Control] Max rounds reached — calling judge.")
        return "judge"                             # force move to judge node

    r1 = state.get("agent1_response", {})          # get all three agents latest responses
    r2 = state.get("agent2_response", {})
    r3 = state.get("agent3_response", {})

    pos1 = r1.get("position", "UNCERTAIN")         # get each agent's current position
    pos2 = r2.get("position", "UNCERTAIN")
    pos3 = r3.get("position", "UNCERTAIN")

    conf_avg = (float(r1.get("confidence", 0.5)) + # calculate average confidence across all 3 agents
                float(r2.get("confidence", 0.5)) +
                float(r3.get("confidence", 0.5))) / 3

    if pos1 == pos2 == pos3 and conf_avg > 0.80:   # if agents reached consensus during debate
        print("  [Control] Consensus reached mid-debate — calling judge.")
        return "judge"                             # no point continuing — go to judge

    return "continue"                              # agents still disagree — keep debating


# =============================================================================
# NODE 5 — Judge: Claude Sonnet reads full debate and gives final verdict
# =============================================================================

def node_judge(state: DebateState) -> dict:
    print("\n  [Judge - Claude Sonnet] Reading full debate and deciding...")

    prompt = f"""News claim: "{state['claim']}"

Full debate transcript:
{state.get('debate_history', '')}

You have read all arguments from all three agents. Give your final verdict."""

    response = judge_call(prompt)                  # send to Claude Sonnet via OpenRouter

    return {                                       # update state with judge's final decision
        "final_verdict":    response.get("verdict",    "INSUFFICIENT_EVIDENCE"),  # default if key missing
        "final_reasoning":  response.get("reasoning",  "No reasoning provided."), # default if key missing
        "final_confidence": float(response.get("confidence", 0.5)),               # default 0.5 if key missing
    }


# =============================================================================
# BUILD THE GRAPH — wire all nodes together into a flowchart
# =============================================================================

def build_graph():
    graph = StateGraph(DebateState)                # create new graph using DebateState as shared notebook

    # register every node — first argument is the name, second is the function to call
    graph.add_node("agent1",       node_agent1)    # Agent 1 node
    graph.add_node("agent2",       node_agent2)    # Agent 2 node
    graph.add_node("agent3",       node_agent3)    # Agent 3 node
    graph.add_node("debate_round", node_debate_round)  # Debate round node
    graph.add_node("judge",        node_judge)     # Judge node

    # fixed edges — these always go in one direction no matter what
    graph.add_edge(START,    "agent1")             # START → agent1 always
    graph.add_edge("agent1", "agent2")             # agent1 → agent2 always
    graph.add_edge("agent2", "agent3")             # agent2 → agent3 always

    # conditional edge after agent3 — goes to judge OR debate_round based on confidence_gate()
    graph.add_conditional_edges(
        "agent3",                                  # from this node
        confidence_gate,                           # call this function to decide direction
        {"skip": "judge", "debate": "debate_round"}  # "skip" → judge, "debate" → debate_round
    )

    # conditional edge after debate_round — loop back OR move to judge based on should_continue()
    graph.add_conditional_edges(
        "debate_round",                            # from this node
        should_continue,                           # call this function to decide direction
        {"continue": "debate_round", "judge": "judge"}  # loop back or move forward
    )

    graph.add_edge("judge", END)                   # judge → END always — judge is always last step

    return graph.compile()                         # compile = finalize and validate the entire graph


# =============================================================================
# MAIN ENTRY POINT — called from main.py and evaluate.py
# =============================================================================

def verify_claim(claim: str) -> dict:              # takes a news claim string, returns full result as dict
    app = build_graph()                            # build and compile the LangGraph flowchart

    initial_state = {                              # starting state — everything empty except the claim
        "claim":            claim,                 # the claim we want to verify
        "round_number":     0,                     # zero debate rounds so far
        "agent1_response":  {},                    # empty — node_agent1 will fill this
        "agent2_response":  {},                    # empty — node_agent2 will fill this
        "agent3_response":  {},                    # empty — node_agent3 will fill this
        "debate_history":   "",                    # empty string — agents will append to this
        "final_verdict":    "",                    # empty — judge will fill this
        "final_reasoning":  "",                    # empty — judge will fill this
        "final_confidence": 0.0,                   # zero — judge will fill this
    }

    result = app.invoke(initial_state)             # run the entire graph — triggers the whole debate pipeline

    return {                                       # return clean summary dict for main.py and evaluate.py
        "claim":          claim,                   # original claim
        "verdict":        result["final_verdict"],         # final decision
        "confidence":     result["final_confidence"],      # judge's confidence score
        "reasoning":      result["final_reasoning"],       # judge's explanation
        "debate_rounds":  result["round_number"],          # how many debate rounds happened
        "debate_history": result["debate_history"],        # full debate transcript
    }