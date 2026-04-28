import os                                          
from dotenv import load_dotenv                     

load_dotenv()                                      


OPENROUTER_API_KEY  = os.getenv("OPENROUTER_API_KEY", "")  
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"         



AGENT1_MODEL = "openai/gpt-4o-mini"              
AGENT2_MODEL = "x-ai/grok-3-mini-beta"    
AGENT3_MODEL = "google/gemini-2.5-flash"          
JUDGE_MODEL  = "anthropic/claude-sonnet-4-5"      



MAX_DEBATE_ROUNDS    = 2     
CONFIDENCE_THRESHOLD = 0.80  