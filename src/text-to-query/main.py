import json
import os
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI
import re

# Load environment variables (.env)
load_dotenv()

app = FastAPI(title="Text-to-SPARQL API")

QLEVER_URL = "http://localhost:7070"

# ---------------------------------------------------------
# CAMPUS AI SETUP
# ---------------------------------------------------------
CAMPUS_API_KEY = os.getenv("CAMPUS_API_KEY")
CAMPUS_API_URL = os.getenv("CAMPUS_API_URL", "https://chat.campusai.compute.dtu.dk/api")
LLM_MODEL_NAME = os.getenv("CAMPUS_MODEL", "Gemma 3 (Chat)") 

llm_client = OpenAI(
    api_key=CAMPUS_API_KEY, 
    base_url=CAMPUS_API_URL
)
class QueryInput(BaseModel):
    text: str

# ---------------------------------------------------------
# CORE SPARQL EXECUTION
# ---------------------------------------------------------
def run_sparql(query: str) -> dict:
    headers = {"Accept": "application/sparql-results+json"}
    with httpx.Client() as client:
        response = client.post(QLEVER_URL, data={"query": query}, headers=headers)
        if response.status_code != 200:
            print(f"QLever Error: {response.text}")
            raise HTTPException(status_code=500, detail="SPARQL execution failed")
        return response.json()

# ---------------------------------------------------------
# GRAPH LOOKUP TOOLS (Updated for Case-Insensitivity)
# ---------------------------------------------------------
def lookup_item(label: str) -> str:
    # Take only the first line and strip it
    label = label.split("\n")[0].strip()
    
    query = f"""
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
    SELECT ?item WHERE {{
      ?item rdfs:label | skos:altLabel ?lbl .
      FILTER(REGEX(REPLACE(STR(?lbl), "[- ]", ""), REPLACE("{label}", "[- ]", ""), "i"))
    }} LIMIT 1
    """
    res = run_sparql(query)
    bindings = res.get("results", {}).get("bindings", [])
    if not bindings:
        return None
    uri = bindings[0]["item"]["value"]
    qid = uri.split("/")[-1]
    return f"kb:{qid}"

def lookup_property(label: str) -> str:
    # Take only the first line and strip it
    label = label.split("\n")[0].strip()
    
    query = f"""
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
    SELECT ?property WHERE {{
      ?property_item rdfs:label | skos:altLabel ?lbl ;
                     <http://wikiba.se/ontology#directClaim> ?property .
      FILTER(LCASE(STR(?lbl)) = LCASE("{label}"))
    }} LIMIT 1
    """
    res = run_sparql(query)
    bindings = res.get("results", {}).get("bindings", [])
    if not bindings:
        return None
    uri = bindings[0]["property"]["value"]
    pid = uri.split("/")[-1]
    return f"kbt:{pid}"

# ---------------------------------------------------------
# REACT AGENT LOOP
# ---------------------------------------------------------
SYSTEM_PROMPT = """
You are an intelligent agent querying a Keyboards Wikibase. 

You have three tools:
1. lookup_item: Finds the QID for a brand or country (e.g., "Yamaha", "Japan").
2. lookup_property: Finds the PID for an attribute (e.g., "width", "manufacturer", "country").
3. run_sparql: Executes the final SPARQL query.

CRITICAL RULES:
- NEVER guess ANY entity or property IDs. You MUST use tools to find them.
- Do NOT use placeholders like `kbt:P_manufacturer`. Find the real PIDs!
- SCHEMA: Keyboards connect to a "manufacturer". The manufacturer connects to a "country".

Here is an example of the exact workflow you must follow for complex queries:

User: What is the biggest Swedish keyboard?
Thought: I need to find the QID for Sweden.
Action: lookup_item
Input: Sweden
Observation: kb:Q34

Thought: I need the property for "country" to link the brand to Sweden.
Action: lookup_property
Input: country
Observation: kbt:P19

Thought: I need the property for "manufacturer" to link the keyboard to the brand.
Action: lookup_property
Input: manufacturer
Observation: kbt:P1

Thought: I need the property for "width" to find the biggest.
Action: lookup_property
Input: width
Observation: kbt:P2

Thought: I have all IDs. I will write a 2-hop SPARQL query ordering by width descending.
Action: run_sparql
Input: 
SELECT ?keyboard WHERE { 
  ?keyboard kbt:P1 ?brand . 
  ?brand kbt:P19 kb:Q34 . 
  ?keyboard kbt:P2 ?width . 
} ORDER BY DESC(?width) LIMIT 1
"""

def execute_react_loop(question: str) -> dict:
    """Runs the ReAct Agent loop until it calls 'run_sparql' or hits the iteration limit."""
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question}
    ]
    
    # Safety net: Max 10 iterations to prevent infinite loops
    for step in range(10):
        # 1. Ask the LLM what to do
        response = llm_client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=messages,
            temperature=0.0, # Deterministic logic
            stop=["Observation:"]
        )
        
        llm_text = response.choices[0].message.content.strip()
        print(f"\n--- Step {step + 1} ---")
        print(llm_text)
        
        # Add the LLM's thought/action to the chat history so it remembers
        messages.append({"role": "assistant", "content": llm_text})
        
        # 2. Parse the Action and Input using Regex
        action_match = re.search(r"Action:\s*(.*)", llm_text)
        # re.DOTALL allows the Input to span multiple lines (crucial for SPARQL queries!)
        input_match = re.search(r"Input:\s*(.*)", llm_text, re.DOTALL) 
        
        if not action_match or not input_match:
            observation = "Observation: Formatting error. Please include both 'Action:' and 'Input:' on separate lines."
            print(f"-> {observation}")
            messages.append({"role": "user", "content": observation})
            continue
            
        action = action_match.group(1).strip()
        action_input = input_match.group(1).strip()
        
        # 3. Execute the corresponding Python Tool
        observation = ""
        if action == "lookup_item":
            res = lookup_item(action_input)
            observation = f"Observation: {res}" if res else f"Observation: Item '{action_input}' not found. Try a synonym or broader category."
            
        elif action == "lookup_property":
            res = lookup_property(action_input)
            observation = f"Observation: {res}" if res else f"Observation: Property '{action_input}' not found. Try a synonym."
            
        elif action == "run_sparql":
            # Clean up markdown if the LLM added it (e.g., ```sparql ... ```)
            clean_sparql = action_input.replace("```sparql", "").replace("```", "").strip()
            
            # FAIL-SAFE FIX: Auto-inject prefixes if the LLM forgot them
            if "PREFIX kb:" not in clean_sparql:
                prefixes = "PREFIX kb: <https://keyboards.wikibase.cloud/entity/>\nPREFIX kbt: <https://keyboards.wikibase.cloud/prop/direct/>\n"
                clean_sparql = prefixes + clean_sparql
            
            # Execute the final query
            try:
                raw_results = run_sparql(clean_sparql)
                bindings = raw_results.get("results", {}).get("bindings", [])
                
                # We format the final output and break the loop
                return {
                    "query": question,
                    "sparql": clean_sparql,
                    "results": bindings,
                    "steps_taken": step + 1
                }
            except Exception as e:
                # TOKEN DIET FIX: Truncate massive QLever errors
                error_msg = str(e)
                if len(error_msg) > 300:
                    error_msg = error_msg[:300] + "... [TRUNCATED]"
                    
                observation = f"Observation: SPARQL execution failed. Error: {error_msg}. Please fix the syntax."
        else:
            observation = f"Observation: Unknown action '{action}'. You must use lookup_item, lookup_property, or run_sparql."
            
        # 4. Feed the observation back to the LLM for the next iteration
        print(f"-> {observation}")
        messages.append({"role": "user", "content": observation})
        
    # If the loop finishes without returning, the agent failed.
    raise HTTPException(status_code=500, detail="Agent exceeded maximum steps without calling run_sparql.")

# ---------------------------------------------------------
# UPDATED ENDPOINT
# ---------------------------------------------------------
@app.post("/v1/query")
def query_endpoint(q: QueryInput):
    # The endpoint is now incredibly clean because the Agent handles all the complexity!
    return execute_react_loop(q.text)