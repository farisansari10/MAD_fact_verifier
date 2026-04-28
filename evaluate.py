import json                                       
import time                                     
from debate_graph import verify_claim              


def load_claims(filepath: str = "claims.json") -> list:
    

    with open(filepath, "r") as f:               
        claims = json.load(f)                     

    print(f"  Loaded {len(claims)} claims from {filepath}\n") 
    return claims                            


def run_evaluation():
    results    = []                             
    correct    = 0                                
    total_time = 0                            

    print("=" * 60)                            
    print("  Multi-Agent Debate — Evaluation Run") 
    print("=" * 60)                               

    TEST_CLAIMS = load_claims("claims.json")       

    for i, test in enumerate(TEST_CLAIMS):      
        print(f"\nClaim {i+1}/{len(TEST_CLAIMS)}: {test['claim']}")  
        print(f"Expected : {test['label']}")       

        start   = time.time()                     
        output  = verify_claim(test["claim"])      
        elapsed = round(time.time() - start, 1)  
        total_time += elapsed                     

        predicted  = output["verdict"]          
        is_correct = (predicted == test["label"]) 

        if is_correct:                            
            correct += 1                          

        print(f"Predicted: {predicted}  (confidence: {output['confidence']:.2f})") 
        print(f"Rounds   : {output['debate_rounds']}  |  Time: {elapsed}s")        
        print(f"Result   : {'✓ CORRECT' if is_correct else '✗ WRONG'}")            

        results.append({                          
            "claim":         test["claim"],     
            "expected":      test["label"],       
            "predicted":     predicted,           
            "confidence":    output["confidence"],
            "correct":       is_correct,          
            "debate_rounds": output["debate_rounds"],  
            "reasoning":     output["reasoning"],  
            "time_seconds":  elapsed,              
        })

   

    accuracy   = correct / len(TEST_CLAIMS) * 100                          
    avg_conf   = sum(r["confidence"] for r in results) / len(results)     
    avg_rounds = sum(r["debate_rounds"] for r in results) / len(results)  

    print("\n" + "=" * 60)
    print("  EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Accuracy          : {accuracy:.1f}%  ({correct}/{len(TEST_CLAIMS)})") 
    print(f"  Avg Confidence    : {avg_conf:.2f}")                                
    print(f"  Avg Debate Rounds : {avg_rounds:.1f}")                                
    print(f"  Total Time        : {total_time:.0f}s")                             
    print("=" * 60)

   

    output_data = {                                
        "accuracy":          accuracy,            
        "avg_confidence":    avg_conf,            
        "avg_debate_rounds": avg_rounds,           
        "total_time":        total_time,           
        "results":           results,              
    }

    with open("evaluation_results.json", "w") as f:  
        json.dump(output_data, f, indent=2, ensure_ascii=False)        

    print("\n  Results saved to evaluation_results.json")  
    print("  Use these numbers in your IEEE paper.\n")     


if __name__ == "__main__":                         
    run_evaluation()                              