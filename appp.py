import os

import openai

import requests

import hashlib # Imported but not used in the provided code, keeping for completeness

import json

import logging

import random

import datetime

import base64 # Added for base64 encoding/decoding



from flask import Flask, request, jsonify

from functools import wraps



# Imports for Algorand SDK

# Ensure py-algorand-sdk is installed (via requirements.txt)

from algosdk.v2client import algod

from algosdk import account, mnemonic # mnemonic and account are imported but not used in provided code



# Imports for TinyDB

# Ensure tinydb is installed (via requirements.txt)

from tinydb import TinyDB, Query



# ==== INITIAL SETUP ====

app = Flask(__name__)



# === APPLICATION START TIME ===

# Define start_time here for the /system/pulse endpoint

start_time = datetime.datetime.utcnow()



# === ENVIRONMENT CONFIG ===

# It's crucial to set these environment variables before running the app.

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

ALGOD_NODE = os.getenv("ALGOD_NODE", 'https://testnet-api.algonode.cloud') # Default to testnet if not set

ALGOD_TOKEN = os.getenv("ALGOD_TOKEN") # Note: Public Algorand nodes often don't require a token for read operations



# === API CONFIG ===

openai.api_key = OPENAI_API_KEY



# === LOGGING CONFIG ===

logging.basicConfig(

filename='neuronexus.log',

level=logging.INFO,

format='%(asctime)s [%(levelname)s] %(message)s'

)



def log_event(event, meta=None):

"""Logs an event with optional metadata."""

logging.info(f"{event} | Meta: {meta if meta else '{}'}")



# === UTILITIES ===

def error_response(message, status_code=400):

"""Generates a standardized JSON error response."""

return jsonify({"error": message}), status_code



def require_json(f):

"""Decorator to ensure requests have a JSON content type."""

@wraps(f)

def decorated_function(*args, **kwargs):

if not request.is_json:

return error_response("Request must be JSON", 415) # 415 Unsupported Media Type

return f(*args, **kwargs)

return decorated_function



def ai_completion(prompt, role="system", temperature=0.85, tokens=450):

"""

Calls the OpenAI Chat Completion API.

Includes basic error handling for API calls.

"""

if not openai.api_key:

log_event("AI Completion Error", {"reason": "OpenAI API key not set"})

raise ValueError("OpenAI API key is not configured.")

try:

response = openai.ChatCompletion.create(

model="gpt-4", # Ensure you have access to gpt-4 or change to a different model like gpt-3.5-turbo

messages=[

{"role": "system", "content": role},

{"role": "user", "content": prompt}

],

temperature=temperature,

max_tokens=tokens

)

return response["choices"][0]["message"]["content"]

except openai.error.OpenAIError as e:

log_event("OpenAI API Error", {"error": str(e), "prompt_len": len(prompt)})

raise RuntimeError(f"OpenAI API call failed: {e}")

except Exception as e:

log_event("AI Completion Unexpected Error", {"error": str(e), "prompt_len": len(prompt)})

raise RuntimeError(f"An unexpected error occurred during AI completion: {e}")



# =========== AI AGENTS ============

class FinanceAgent:

"""An AI financial coach."""

role = "You are an expert AI financial coach. Help the user optimize personal finances, savings, budgeting, and investing."



def respond(self, message):

log_event("FinanceAgent Invoked", {"query": message})

return ai_completion(message, role=self.role)



class MindAgent:

"""A compassionate AI mental health coach."""

role = "You are a compassionate AI mental health coach. Offer support, mindset shifts, and emotional clarity using science-backed methods."



def respond(self, message):

log_event("MindAgent Invoked", {"query": message})

return ai_completion(message, role=self.role)



class CreativeAgent:

"""An AI creative director for content generation."""

role = "You are an AI creative director. Generate witty Reddit posts, inspiring article ideas, and viral meme captions."



def reddit_post(self, topic="self-improvement"):

prompt = f"Write a Reddit post title and body for r/{topic}. Be clever, humanlike, and on-theme."

post = ai_completion(prompt, role=self.role)

parts = post.split("\n", 1)

return {

"title": parts[0].strip(),

"body": parts[1].strip() if len(parts) > 1 else "(No body)"

}



def meme_caption(self, topic="AI"):

prompt = f"Write a funny, short meme caption about {topic}, maximum 20 words."

return ai_completion(prompt, role=self.role)



def blog_idea(self, niche="technology + mindfulness"):

prompt = f"Generate a thought-provoking blog article title about {niche}."

return ai_completion(prompt, role=self.role)



class VideoCoach:

"""Placeholder for a video coaching simulation."""

def simulate_session(self, prompt, tokens=450): # Added tokens parameter

reply = ai_completion(f"As a video coach, respond to: '{prompt}' with confident language & interview feedback.", tokens=tokens)

return f"🎥 Tavus-style AI says: {reply}\n\n(Note: This is a placeholder. Future integration with Tavus API.)"



class RewriterAgent:

"""An AI prompt optimizer."""

role = "You are a prompt-optimizer AI that rewrites vague or confusing queries into clear, precise language."



def refine(self, raw_prompt):

prompt = f"Rewrite the following prompt for an AI coach: '{raw_prompt}'"

return ai_completion(prompt, role=self.role, tokens=100)



class PostProcessor:

"""Performs post-processing on AI generated text."""

def tag_keywords(self, text):

keywords = []

for word in ["debt", "startup", "job", "anxiety", "invest", "Reddit", "meme", "GPT"]:

if word.lower() in text.lower():

keywords.append(word)

return keywords



def sentiment_score(self, text):

prompt = f"What is the sentiment of this text? '{text}'. Answer with one word: Positive, Neutral, or Negative."

result = ai_completion(prompt, role="Sentiment evaluator.", tokens=5)

return result.strip()



# Instance Initialization of Agents

finance_agent = FinanceAgent()

mind_agent = MindAgent()

creative_agent = CreativeAgent()

video_coach = VideoCoach()

rewriter_agent = RewriterAgent()

post_processor = PostProcessor()



# ========== VOICE INTERFACE ==========

def text_to_speech(text, voice="Rachel"):

"""

Converts text to speech using ElevenLabs API.

Includes error handling.

"""

if not ELEVENLABS_API_KEY:

log_event("Text-to-Speech Error", {"reason": "ElevenLabs API key not set"})

raise ValueError("ElevenLabs API key is not configured.")

try:

url = "https://api.elevenlabs.io/v1/text-to-speech/21m00Tzpb8CxoXLnyC0v" # Defaulting to 'Rachel' voice_id

# Note: You can retrieve a list of available voices from ElevenLabs API

# voice_ids: Adam (pNInz6obpgDQGxUGXP57), Rachel (21m00Tzpb8CxoXLnyC0v), Clyde (jBRPYwA8G2jQzDrTzQ7I)

# Use the voice parameter to select from a predefined list or map

# For simplicity, hardcoding one here or using a default.

# If 'voice' parameter is passed, you'd map it to the correct voice ID.

if voice == "Rachel":

voice_id = "21m00Tzpb8CxoXLnyC0v"

elif voice == "Adam":

voice_id = "pNInz6obpgDQGxUGXP57"

# Add more voice mappings as needed

else:

voice_id = "21m00Tzpb8CxoXLnyC0v" # Default fallback



url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"



headers = {

"xi-api-key": ELEVENLABS_API_KEY,

"Content-Type": "application/json"

}

payload = {

"text": text,

"voice_settings": {"stability": 0.7, "similarity_boost": 0.8}

}

response = requests.post(url, headers=headers, json=payload)

response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)

return response.content

except requests.exceptions.RequestException as e:

log_event("ElevenLabs API Error", {"error": str(e), "text_len": len(text)})

raise RuntimeError(f"ElevenLabs API call failed: {e}")

except Exception as e:

log_event("Text-to-Speech Unexpected Error", {"error": str(e), "text_len": len(text)})

raise RuntimeError(f"An unexpected error occurred during text-to-speech: {e}")





# ========== ALGORAND AUTH ==========

# Initialize Algorand client

algod_client = algod.AlgodClient(ALGOD_TOKEN, ALGOD_NODE)



def verify_algorand_wallet(wallet_address):

"""

Verifies if an Algorand wallet address is valid and exists on the network.

"""

try:

# Note: For public testnet nodes, X-Algo-API-Token is often not required for account_info

# If ALGOD_TOKEN is empty, exclude the header or provide a dummy.

headers = {}

if ALGOD_TOKEN:

headers = {"X-Algo-API-Token": ALGOD_TOKEN}



response = requests.get(f"{ALGOD_NODE}/v2/accounts/{wallet_address}", headers=headers)

response.raise_for_status() # Raise an exception for HTTP errors

return response.status_code == 200

except requests.exceptions.RequestException as e:

log_event("Algorand Verification Failed (Request Error)", {"wallet": wallet_address, "error": str(e)})

return False

except Exception as e:

log_event("Algorand Verification Failed (Unexpected Error)", {"wallet": wallet_address, "error": str(e)})

return False



def get_account_balance(address):

"""Fetches the balance of an Algorand account."""

try:

headers = {}

if ALGOD_TOKEN:

headers = {"X-Algo-API-Token": ALGOD_TOKEN}

acct_info = algod_client.account_info(address, headers=headers) # Pass headers to account_info

return acct_info.get("amount", 0)

except Exception as e:

log_event("Algorand Balance Fetch Error", {"error": str(e)})

return 0



# ========== LIGHTWEIGHT LOCAL DATASTORE (In-memory) ==========

class DataStore:

"""A simple in-memory key-value store for user data."""

memory = {} # This will reset on app restart



def remember(self, user_id, key, value):

"""Stores a value associated with a user and key."""

if user_id not in self.memory:

self.memory[user_id] = {}

self.memory[user_id][key] = value

log_event("MemoryUpdate", {"user": user_id, "key": key, "value": value})



def recall(self, user_id, key):

"""Recalls a value for a given user and key."""

return self.memory.get(user_id, {}).get(key, None)



datastore = DataStore()



# ========== PERSISTENT LOCAL DATASTORE (TinyDB) ==========

# Initialize TinyDB database. It will create 'nexus_memory.json' file

# in the directory where app.py is run.

db_persistent = TinyDB("nexus_memory.json")



# ========== USER REGISTRATION / AUTH (In-memory) ==========

users = {} # This dictionary stores user registration in-memory and will reset on app restart



def generate_token(length=32):

"""Generates a random alphanumeric token."""

alphabet = string.ascii_letters + string.digits

return ''.join(secrets.choice(alphabet) for _ in range(length))



# ========= THREAD CONTEXT MANAGER (In-memory) ==========

thread_contexts = {} # This dictionary stores conversation history in-memory and will reset on app restart



def update_thread(user_id, message):

"""Updates the conversation thread for a given user."""

if user_id not in thread_contexts:

thread_contexts[user_id] = []

thread_contexts[user_id].append({

"timestamp": datetime.datetime.utcnow().isoformat(),

"message": message

})

if len(thread_contexts[user_id]) > 20: # Keep thread concise

thread_contexts[user_id] = thread_contexts[user_id][-20:]



def get_thread_summary(user_id):

"""Summarizes the conversation thread for a given user using AI."""

messages = thread_contexts.get(user_id, [])

history = "\n".join([f"{m['timestamp']}: {m['message']}" for m in messages])

if not history:

return "No prior context."

return ai_completion(f"Summarize this chat log: {history}", tokens=100, role="Conversation memory summarizer.")



# ========= TASK QUEUE ENGINE ==========

class Task:

"""Represents a single task in a workflow."""

def __init__(self, name, agent, prompt, postprocess=False, speak=False, voice="Rachel"):

self.name = name

self.agent = agent

self.prompt = prompt

self.postprocess = postprocess

self.speak = speak

self.voice = voice # Added voice parameter

self.output = None

self.tags = []

self.sentiment = None # Added sentiment

self.audio = None



def run(self):

"""Executes the task using its assigned agent."""

log_event("Task Running", {"task_name": self.name, "agent": self.agent.__class__.__name__})

try:

if isinstance(self.agent, VideoCoach):

self.output = self.agent.simulate_session(self.prompt)

elif hasattr(self.agent, 'respond'): # Generic check for respond method

self.output = self.agent.respond(self.prompt)

elif hasattr(self.agent, 'reddit_post') and "reddit" in self.name.lower():

self.output = self.agent.reddit_post(self.prompt) # Example: if prompt is just a topic

elif hasattr(self.agent, 'meme_caption') and "meme" in self.name.lower():

self.output = self.agent.meme_caption(self.prompt)

elif hasattr(self.agent, 'blog_idea') and "blog" in self.name.lower():

self.output = self.agent.blog_idea(self.prompt)

elif hasattr(self.agent, 'refine') and "rewrite" in self.name.lower():

self.output = self.agent.refine(self.prompt)

else:

# Fallback for agents without a standard 'respond' method or specific task methods

self.output = ai_completion(self.prompt, role=getattr(self.agent, 'role', 'assistant'))



if self.postprocess:

self.tags = post_processor.tag_keywords(self.output)

self.sentiment = post_processor.sentiment_score(self.output) # Get sentiment



if self.speak:

audio_content = text_to_speech(self.output, voice=self.voice)

self.audio = base64.b64encode(audio_content).decode("utf-8")

except Exception as e:

log_event("Task Failed", {"task_name": self.name, "error": str(e)})

self.output = f"Error executing task '{self.name}': {str(e)}"

self.audio = None # Ensure audio is not attempted if error occurred

return self



class Workflow:

"""Represents a sequence of tasks to be executed."""

def __init__(self, name):

self.name = name

self.tasks = []



def add_task(self, task: Task):

"""Adds a task to the workflow."""

self.tasks.append(task)



def run_all(self):

"""Runs all tasks in the workflow sequentially."""

results = []

for task in self.tasks:

results.append(task.run().__dict__) # Returns the task's attributes as a dict

return results



# ========== SENTIENT CHAIN EXECUTOR (Multi-Agent Orchestrator) ==========

class SentientChainExecutor:

"""

Orchestrates multi-agent interactions, leveraging memory, agent routing,

and prompt optimization for complex user queries.

"""

def __init__(self, memory_store, agent_router, rewriter_agent, post_processor, thread_contexts_ref):

"""

Initializes the SentientChainExecutor with necessary components.



Args:

memory_store (DataStore): An instance of the local data store for memory recall/save.

agent_router (Router): An instance of the Router to select appropriate agents.

rewriter_agent (RewriterAgent): An instance of the RewriterAgent for prompt optimization.

post_processor (PostProcessor): An instance of the PostProcessor for output analysis.

thread_contexts_ref (dict): A mutable reference to the global thread_contexts dictionary.

"""

self.memory = memory_store

self.router = agent_router

self.rewriter = rewriter_agent

self.post_processor = post_processor

self.thread_contexts = thread_contexts_ref # Store reference to the global dict



def _get_thread_summary_internal(self, user_id):

"""Internal helper to get summarized thread context using the global function."""

return get_thread_summary(user_id)



def _update_thread_internal(self, user_id, message):

"""Internal helper to update the live thread context using the global function."""

update_thread(user_id, message)



def execute_chain(self, user_id, raw_prompt, persist_memory=False, speak_output=False, voice="Rachel"):

"""

Executes a multi-stage AI chain based on a user prompt, utilizing various agents.



Args:

user_id (str): Identifier for the user, for memory and thread context.

raw_prompt (str): The initial user prompt.

persist_memory (bool): Whether to save the interaction to persistent memory.

speak_output (bool): Whether to convert the final response to speech.

voice (str): The voice to use for speech output (e.g., "Rachel", "Adam").



Returns:

dict: A dictionary containing the processed output, agent used, tags, sentiment, and audio (if requested).

"""

log_event("Chain Execution Initiated", {"user_id": user_id, "prompt": raw_prompt})

self._update_thread_internal(user_id, f"User: {raw_prompt}")



# 1. Get current context summary

context_summary = self._get_thread_summary_internal(user_id)

logging.info(f"Context for {user_id}: {context_summary}")



# 2. Refine the prompt using RewriterAgent

refined_prompt = self.rewriter.refine(raw_prompt)

logging.info(f"Refined prompt: {refined_prompt}")



# 3. Route to the appropriate agent

selected_agent = self.router.route(refined_prompt)

agent_name = selected_agent.__class__.__name__

logging.info(f"Routed to agent: {agent_name}")



# 4. Get response from the selected agent

agent_response = ""

try:

# Task handles the specific agent method call

temp_task = Task("ChainExecutionResponse", selected_agent, refined_prompt,

postprocess=True, speak=speak_output, voice=voice)

temp_task.run() # Execute the task

agent_response = temp_task.output

sentiment = temp_task.sentiment

tags = temp_task.tags

audio_base64 = temp_task.audio



except Exception as e:

log_event("Sentient Chain Agent Response Error", {"error": str(e), "user_id": user_id, "agent": agent_name})

agent_response = f"Error getting response from {agent_name}: {str(e)}"

sentiment = "Negative"

tags = ["error"]

audio_base64 = None



self._update_thread_internal(user_id, f"Agent ({agent_name}): {agent_response}")

log_event("Agent Response Generated", {"agent": agent_name, "response_len": len(agent_response)})



# 5. Optionally save to persistent memory (using TinyDB)

if persist_memory:

db_persistent.insert({

"user": user_id,

"key": "last_interaction", # You could make this key dynamic

"value": {

"prompt": raw_prompt,

"response": agent_response,

"agent": agent_name,

"timestamp": datetime.datetime.utcnow().isoformat(),

"sentiment": sentiment,

"keywords": tags

},

"ts": datetime.datetime.utcnow().isoformat()

})

log_event("Interaction Saved to Persistent TinyDB", {"user_id": user_id})



return {

"user_id": user_id,

"raw_prompt": raw_prompt,

"refined_prompt": refined_prompt,

"context_summary": context_summary,

"agent_used": agent_name,

"final_response": agent_response,

"sentiment": sentiment,

"keywords": tags,

"audio_mpeg_base64": audio_base64

}





# Initialize the SentientChainExecutor after all its dependencies

# (datastore, router, rewriter_agent, post_processor, thread_contexts) are defined.

class Router:

"""Routes prompts to the appropriate AI agent."""

def route(self, prompt):

lowered = prompt.lower()

if any(word in lowered for word in ["save", "invest", "money", "debt", "finance", "budget"]):

return finance_agent

elif any(word in lowered for word in ["anxious", "sad", "confidence", "stress", "overwhelmed", "mood", "mental"]):

return mind_agent

elif any(word in lowered for word in ["post", "reddit", "caption", "meme", "funny", "creative", "blog", "idea"]):

return creative_agent

elif any(word in lowered for word in ["job", "career", "interview", "resume", "pitch"]):

return video_coach

else:

return finance_agent # Fallback



router = Router() # Instantiate the router after all agents are defined



sentient_executor = SentientChainExecutor(

memory_store=datastore, # In-memory DataStore

agent_router=router,

rewriter_agent=rewriter_agent,

post_processor=post_processor,

thread_contexts_ref=thread_contexts # Pass the global thread_contexts dict by reference

)





# ========== MINI ROUTES FOR HACK FUN ==========

@app.route("/")

def root():

return jsonify({"status": "Neuronexus AI Engine operational", "agents": ["Finance", "Mind", "Creative", "VideoCoach", "Rewriter"]})



@app.route("/finance", methods=["POST"])

@require_json

def route_finance():

data = request.get_json()

query = data.get("prompt")

if not query:

return error_response("Missing prompt.")

response = finance_agent.respond(query)

return jsonify({"reply": response})



@app.route("/mind", methods=["POST"])

@require_json

def route_mind():

data = request.get_json()

query = data.get("prompt")

if not query:

return error_response("Missing prompt.")

response = mind_agent.respond(query)

return jsonify({"reply": response})



@app.route("/speak", methods=["POST"])

@require_json

def voice_output():

data = request.get_json()

text = data.get("text")

voice = data.get("voice", "Rachel") # Allow specifying voice

if not text:

return error_response("Missing text.")

try:

audio = text_to_speech(text, voice=voice)

return audio, 200, {"Content-Type": "audio/mpeg"}

except RuntimeError as e:

return error_response(str(e), 500)



@app.route("/verify", methods=["POST"])

@require_json

def wallet_verify():

data = request.get_json()

wallet = data.get("wallet")

if not wallet:

return error_response("Missing wallet address.")

valid = verify_algorand_wallet(wallet)

return jsonify({"verified": valid})



@app.route("/madlib", methods=["GET"])

def madlib_game():

prompts = [

"The cat philosophically debated a cactus before teleporting.",

"In the middle of traffic, the toaster levitated and sang jazz.",

"Everyone was shocked when the robot cried over pineapple pizza."

]

return jsonify({"madlib": random.choice(prompts)})



@app.route("/status", methods=["GET"])

def status_check():

now = datetime.datetime.utcnow().isoformat() + "Z"

return jsonify({

"status": "Neuronexus is live",

"timestamp": now,

"version": "0.1a"

})



@app.route("/creative/reddit", methods=["GET"])

def creative_reddit():

topic = request.args.get("topic", "self-improvement")

post = creative_agent.reddit_post(topic)

return jsonify(post)



@app.route("/creative/meme", methods=["GET"])

def creative_meme():

topic = request.args.get("topic", "AI")

caption = creative_agent.meme_caption(topic)

return jsonify({"meme_caption": caption})



@app.route("/creative/blog", methods=["GET"])

def creative_blog():

niche = request.args.get("niche", "technology + mindfulness")

idea = creative_agent.blog_idea(niche)

return jsonify({"blog_idea": idea})





@app.route("/video/coach", methods=["POST"])

@require_json

def video_session():

data = request.get_json()

prompt = data.get("prompt", "")

if not prompt:

return error_response("Missing prompt for video simulation.")

result = video_coach.simulate_session(prompt)

return jsonify({"video_response": result})





# ========= SUBSCRIPTION TIER SIM (REVENUECAT STUB) ==========

USER_SUBSCRIPTIONS = { # This is an in-memory stub

"0x123fakewallet": "premium",

"0x999noobwallet": "free"

}



@app.route("/subscription/status", methods=["POST"])

@require_json

def subscription_status():

data = request.get_json()

wallet = data.get("wallet", "")

tier = USER_SUBSCRIPTIONS.get(wallet, "free")

return jsonify({"wallet": wallet, "tier": tier})





@app.route("/memory/save", methods=["POST"])

@require_json

def memory_save():

data = request.get_json()

user = data.get("user_id")

key = data.get("key")

val = data.get("value")

if not all([user, key, val]):

return error_response("Missing parameters.")

datastore.remember(user, key, val)

return jsonify({"status": "Saved."})



@app.route("/memory/load", methods=["POST"])

@require_json

def memory_load():

data = request.get_json()

user = data.get("user_id")

key = data.get("key")

if not all([user, key]):

return error_response("Missing parameters.")

val = datastore.recall(user, key)
    return jsonify({"value": val})

@app.route("/memory/persistent/save", methods=["POST"])
@require_json
def persistent_memory_save():
    data = request.get_json()
    user_id = data.get("user_id")
    key = data.get("key")
    value = data.get("value")
    if not all([user_id, key, value]):
        return error_response("Missing user_id, key, or value.")
    try:
        db_persistent.insert({"user_id": user_id, "key": key, "value": value, "timestamp": datetime.datetime.utcnow().isoformat()})
        log_event("PersistentMemorySave", {"user_id": user_id, "key": key})
        return jsonify({"status": "Data saved to persistent memory."})
    except Exception as e:
        log_event("PersistentMemorySaveError", {"user_id": user_id, "key": key, "error": str(e)})
        return error_response(f"Error saving to persistent memory: {str(e)}", 500)

@app.route("/memory/persistent/load", methods=["POST"])
@require_json
def persistent_memory_load():
    data = request.get_json()
    user_id = data.get("user_id")
    key = data.get("key")
    if not all([user_id, key]):
        return error_response("Missing user_id or key.")
    try:
        UserQuery = Query()
        result = db_persistent.search((UserQuery.user_id == user_id) & (UserQuery.key == key))
        if result:
            log_event("PersistentMemoryLoad", {"user_id": user_id, "key": key, "found": True})
            return jsonify({"value": result[-1].get("value")}) # Return the most recent entry
        else:
            log_event("PersistentMemoryLoad", {"user_id": user_id, "key": key, "found": False})
            return jsonify({"value": None, "message": "No data found for this user and key."})
    except Exception as e:
        log_event("PersistentMemoryLoadError", {"user_id": user_id, "key": key, "error": str(e)})
        return error_response(f"Error loading from persistent memory: {str(e)}", 500)


@app.route("/system/pulse", methods=["GET"])
def system_pulse():
    """Provides a heartbeat and uptime of the system."""
    uptime_seconds = (datetime.datetime.utcnow() - start_time).total_seconds()
    hours, remainder = divmod(uptime_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    uptime_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
    return jsonify({
        "status": "online",
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "uptime": uptime_str
    })

@app.route("/account/balance", methods=["POST"])
@require_json
def account_balance():
    data = request.get_json()
    wallet_address = data.get("wallet_address")
    if not wallet_address:
        return error_response("Missing wallet_address.")
    balance = get_account_balance(wallet_address)
    return jsonify({"wallet_address": wallet_address, "balance": balance})

@app.route("/ai/chain", methods=["POST"])
@require_json
def ai_chain_execute():
    """
    Endpoint for the Sentient Chain Executor to handle complex multi-agent interactions.
    """
    data = request.get_json()
    user_id = data.get("user_id")
    raw_prompt = data.get("prompt")
    persist_memory = data.get("persist_memory", False)
    speak_output = data.get("speak_output", False)
    voice = data.get("voice", "Rachel") # Default voice

    if not all([user_id, raw_prompt]):
        return error_response("Missing user_id or prompt.")

    try:
        result = sentient_executor.execute_chain(user_id, raw_prompt, persist_memory, speak_output, voice)
        return jsonify(result)
    except Exception as e:
        log_event("AIChainExecutionError", {"user_id": user_id, "prompt": raw_prompt, "error": str(e)})
        return error_response(f"An error occurred during AI chain execution: {str(e)}", 500)

@app.route("/ai/thread_context", methods=["POST"])
@require_json
def get_thread_context_endpoint():
    """Endpoint to retrieve a summarized thread context for a user."""
    data = request.get_json()
    user_id = data.get("user_id")
    if not user_id:
        return error_response("Missing user_id.")
    try:
        summary = get_thread_summary(user_id)
        return jsonify({"user_id": user_id, "thread_summary": summary})
    except Exception as e:
        log_event("GetThreadContextError", {"user_id": user_id, "error": str(e)})
        return error_response(f"Error retrieving thread context: {str(e)}", 500)

# The following two imports are needed for 'secrets' and 'string'
import secrets
import string

@app.route("/user/register", methods=["POST"])
@require_json
def register_user():
    """
    Registers a new user (in-memory for this stub).
    In a real application, this would involve a database and proper authentication.
    """
    data = request.get_json()
    username = data.get("username")
    password = data.get("password") # In a real app, hash and salt this!
    wallet_address = data.get("wallet_address") # Optional: link wallet at registration

    if not all([username, password]):
        return error_response("Missing username or password.")

    if username in users:
        return error_response("Username already exists.", 409) # 409 Conflict

    user_id = hashlib.sha256(username.encode()).hexdigest() # Simple user ID generation
    auth_token = generate_token() # Generate a session token

    users[username] = {
        "user_id": user_id,
        "password_hash": hashlib.sha256(password.encode()).hexdigest(), # Storing hash, not plain password
        "auth_token": auth_token,
        "wallet_address": wallet_address,
        "created_at": datetime.datetime.utcnow().isoformat()
    }
    log_event("User Registered", {"username": username, "user_id": user_id, "wallet_linked": bool(wallet_address)})
    return jsonify({
        "message": "User registered successfully",
        "user_id": user_id,
        "auth_token": auth_token,
        "username": username
    }), 201 # 201 Created

@app.route("/user/login", methods=["POST"])
@require_json
def login_user():
    """
    Logs in a user and provides an auth token (in-memory for this stub).
    """
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")

    if not all([username, password]):
        return error_response("Missing username or password.")

    user_info = users.get(username)
    if not user_info:
        return error_response("Invalid username or password.", 401) # 401 Unauthorized

    if user_info["password_hash"] != hashlib.sha256(password.encode()).hexdigest():
        return error_response("Invalid username or password.", 401)

    # Regenerate token on login for better security (or use refresh tokens)
    new_auth_token = generate_token()
    user_info["auth_token"] = new_auth_token
    log_event("User Logged In", {"username": username, "user_id": user_info["user_id"]})

    return jsonify({
        "message": "Login successful",
        "user_id": user_info["user_id"],
        "auth_token": new_auth_token
    })

# --- Main execution ---
if __name__ == "__main__":
    # Ensure environment variables are set before running the app
    if not OPENAI_API_KEY:
        logging.error("OPENAI_API_KEY environment variable not set.")
        # Optionally sys.exit(1) or raise an exception here in a production setup
    if not ELEVENLABS_API_KEY:
        logging.warning("ELEVENLABS_API_KEY environment variable not set. Voice features will be disabled.")
    # ALGOD_TOKEN can be None for public nodes, so no strict check here.

    # Running Flask in debug mode. For production, use a WSGI server like Gunicorn.
    app.run(debug=True, host='0.0.0.0', port=5000)
