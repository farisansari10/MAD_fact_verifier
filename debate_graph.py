import json                                        
from typing import TypedDict                      
from langgraph.graph import StateGraph, START, END 
from config import MAX_DEBATE_ROUNDS, CONFIDENCE_THRESHOLD  
from tools import search_web, search_wikipedia     
from llm_caller import (                           
    agent1_call, agent2_call, agent3_call,        
    agent1_debate, agent2_debate, agent3_debate,   
    judge_call,                                    
)


class DebateState(TypedDict):        
    claim:            str             
    round_number:     int             
    agent1_response:  dict            
    agent2_response:  dict           
    agent3_response:  dict           
    debate_history:   str             
    final_verdict:    str            
    final_reasoning:  str             
    final_confidence: float         


def node_agent1(state: DebateState) -> dict:       
    print("\n  [Agent 1 - GPT-4o-mini] Searching the web...")

    web_results = search_web(state["claim"])       

    
    prompt = f"""News claim to verify: "{state['claim']}"

Web search results:
{web_results}

Based on this evidence, is the claim SUPPORTED, REFUTED, or UNCERTAIN?"""

    response = agent1_call(prompt)                  

    history = state.get("debate_history", "")       
    history += f"\n[ROUND 0 - Agent 1 (GPT-4o-mini)]\n"          
    history += f"Position  : {response.get('position')}\n"        
    history += f"Confidence: {response.get('confidence')}\n"      
    history += f"Reasoning : {response.get('reasoning')}\n"       
    history += f"Evidence  : {response.get('evidence')}\n"        

    return {                                        
        "agent1_response": response,               
        "debate_history":  history,               
    }




def node_agent2(state: DebateState) -> dict:       
    print("  [Agent 2 - Grok 3 Mini Beta] Checking Wikipedia...")

    wiki_results = search_wikipedia(state["claim"]) 

    prompt = f"""News claim to verify: "{state['claim']}"

Wikipedia background:
{wiki_results}

Based on this background information, is the claim SUPPORTED, REFUTED, or UNCERTAIN?"""

    response = agent2_call(prompt)                 

    history = state.get("debate_history", "")      
    history += f"\n[ROUND 0 - Agent 2 (Grok 3 Mini Beta)]\n"
    history += f"Position  : {response.get('position')}\n"
    history += f"Confidence: {response.get('confidence')}\n"
    history += f"Reasoning : {response.get('reasoning')}\n"
    history += f"Evidence  : {response.get('evidence')}\n"

    return {
        "agent2_response": response,               
        "debate_history":  history,               
    }



def node_agent3(state: DebateState) -> dict:       
    print("  [Agent 3 - Gemini 2.5 Flash] Reasoning independently...")

    
    prompt = f"""News claim to verify: "{state['claim']}"

Use only your own knowledge and logical reasoning — no external tools available.
Is this claim SUPPORTED, REFUTED, or UNCERTAIN?"""

    response = agent3_call(prompt)                  

    history = state.get("debate_history", "")     
    history += f"\n[ROUND 0 - Agent 3 (Gemini 2.5 Flash)]\n"
    history += f"Position  : {response.get('position')}\n"
    history += f"Confidence: {response.get('confidence')}\n"
    history += f"Reasoning : {response.get('reasoning')}\n"
    history += f"Evidence  : {response.get('evidence')}\n"

    return {
        "agent3_response": response,               
        "debate_history":  history,               
    }



def confidence_gate(state: DebateState) -> str:    

    r1 = state.get("agent1_response", {})         
    r2 = state.get("agent2_response", {})          
    r3 = state.get("agent3_response", {})          
    pos1 = r1.get("position", "UNCERTAIN")         
    pos2 = r2.get("position", "UNCERTAIN")         
    pos3 = r3.get("position", "UNCERTAIN")         

    conf1 = float(r1.get("confidence", 0.5))       
    conf2 = float(r2.get("confidence", 0.5))      
    conf3 = float(r3.get("confidence", 0.5))       

    all_agree     = (pos1 == pos2 == pos3) and pos1 != "UNCERTAIN"  
    all_high_conf = (conf1 >= CONFIDENCE_THRESHOLD and             
                     conf2 >= CONFIDENCE_THRESHOLD and
                     conf3 >= CONFIDENCE_THRESHOLD)

    if all_agree and all_high_conf:                
        print("  [Gate] All agents agree with high confidence — skipping debate.")
        return "skip"                             

    print("  [Gate] Disagreement or low confidence detected — starting debate.")
    return "debate"                                




def node_debate_round(state: DebateState) -> dict:
    round_num = state.get("round_number", 0) + 1  
    print(f"\n  [Debate Round {round_num}] Agents reviewing each other's arguments...")

    history = state.get("debate_history", "")     

    
    fresh_web = search_web(state["claim"] + " latest news") 

    prompt1 = f"""Claim: "{state['claim']}"

Full debate so far:
{history}

Fresh web search:
{fresh_web}

You are Agent 1. Read the other agents arguments carefully and update your position."""

    r1 = agent1_debate(prompt1)                   
    history += f"\n[ROUND {round_num} - Agent 1 rebuttal]\n"
    history += f"Position  : {r1.get('position')}\n"
    history += f"Confidence: {r1.get('confidence')}\n"
    history += f"Counter   : {r1.get('counter_point')}\n" 

   
    prompt2 = f"""Claim: "{state['claim']}"

Full debate so far:
{history}

You are Agent 2. Read the other agents arguments carefully and update your position."""

    r2 = agent2_debate(prompt2)                
    history += f"\n[ROUND {round_num} - Agent 2 rebuttal]\n"
    history += f"Position  : {r2.get('position')}\n"
    history += f"Confidence: {r2.get('confidence')}\n"
    history += f"Counter   : {r2.get('counter_point')}\n"

    
    prompt3 = f"""Claim: "{state['claim']}"

Full debate so far:
{history}

You are Agent 3. Read the other agents arguments carefully and update your reasoning."""

    r3 = agent3_debate(prompt3)                
    history += f"\n[ROUND {round_num} - Agent 3 rebuttal]\n"
    history += f"Position  : {r3.get('position')}\n"
    history += f"Confidence: {r3.get('confidence')}\n"
    history += f"Counter   : {r3.get('counter_point')}\n"

    return {                                        
        "round_number":    round_num,              
        "agent1_response": r1,                     
        "agent2_response": r2,                     
        "agent3_response": r3,                    
        "debate_history":  history,                
    }


def should_continue(state: DebateState) -> str:
    if state.get("round_number", 0) >= MAX_DEBATE_ROUNDS:  
        print("  [Control] Max rounds reached — calling judge.")
        return "judge"                             

    r1 = state.get("agent1_response", {})          
    r2 = state.get("agent2_response", {})
    r3 = state.get("agent3_response", {})

    pos1 = r1.get("position", "UNCERTAIN")         
    pos2 = r2.get("position", "UNCERTAIN")
    pos3 = r3.get("position", "UNCERTAIN")

    conf_avg = (float(r1.get("confidence", 0.5)) + 
                float(r2.get("confidence", 0.5)) +
                float(r3.get("confidence", 0.5))) / 3

    if pos1 == pos2 == pos3 and conf_avg > 0.80:   
        print("  [Control] Consensus reached mid-debate — calling judge.")
        return "judge"                             

    return "continue"                              



def node_judge(state: DebateState) -> dict:
    print("\n  [Judge - Claude Sonnet] Reading full debate and deciding...")

    prompt = f"""News claim: "{state['claim']}"

Full debate transcript:
{state.get('debate_history', '')}

You have read all arguments from all three agents. Give your final verdict."""

    response = judge_call(prompt)                  

    return {                                      
        "final_verdict":    response.get("verdict",    "INSUFFICIENT_EVIDENCE"),  
        "final_reasoning":  response.get("reasoning",  "No reasoning provided."), 
        "final_confidence": float(response.get("confidence", 0.5)),              
    }


def build_graph():
    graph = StateGraph(DebateState)                

    
    graph.add_node("agent1",       node_agent1)    
    graph.add_node("agent2",       node_agent2)   
    graph.add_node("agent3",       node_agent3)    
    graph.add_node("debate_round", node_debate_round)  
    graph.add_node("judge",        node_judge)    

   
    graph.add_edge(START,    "agent1")            
    graph.add_edge("agent1", "agent2")             
    graph.add_edge("agent2", "agent3")             

    
    graph.add_conditional_edges(
        "agent3",                                  
        confidence_gate,                       
        {"skip": "judge", "debate": "debate_round"}  
    )

    
    graph.add_conditional_edges(
        "debate_round",                            
        should_continue,                           
        {"continue": "debate_round", "judge": "judge"}  
    )

    graph.add_edge("judge", END)                   

    return graph.compile()                      



def verify_claim(claim: str) -> dict:              
    app = build_graph()                           

    initial_state = {                             
        "claim":            claim,               
        "round_number":     0,                    
        "agent1_response":  {},                    
        "agent2_response":  {},                   
        "agent3_response":  {},                    
        "debate_history":   "",                    
        "final_verdict":    "",                    
        "final_reasoning":  "",                    
        "final_confidence": 0.0,                
    }

    result = app.invoke(initial_state)            

    return {                                       
        "claim":          claim,                
        "verdict":        result["final_verdict"],         
        "confidence":     result["final_confidence"],    
        "reasoning":      result["final_reasoning"],      
        "debate_rounds":  result["round_number"],         
        "debate_history": result["debate_history"],       
    }