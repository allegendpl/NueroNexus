# Neuronexus: app.py —  Fullstack Intelligent Agent Engine 

import os
import openai
import requests
import hashlib
import json
import logging
import random
import datetime
from flask import Flask, request, jsonify
from functools import wraps
pip install py-algorand-sdk
pip install tinydb
# ==== INITIAL SETUP ====
app = Flask(__name__)
start_time = datetime.datetime.utcnow().isoformat()
# === ENVIRONMENT CONFIG ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ALGOD_NODE = 'https://testnet-api.algonode.cloud'
ALGOD_TOKEN = os.getenv("ALGOD_TOKEN")

# === API CONFIG ===
openai.api_key = OPENAI_API_KEY

# === LOGGING CONFIG ===
logging.basicConfig(
    filename='neuronexus.log',
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)

def log_event(event, meta=None):
    logging.info(f"{event} | Meta: {meta if meta else '{}'}")

# === UTILITIES ===
def error_response(message):
    return jsonify({"error": message}), 400

def require_json(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not request.is_json:
            return error_response("Request must be JSON")
        return f(*args, **kwargs)
    return decorated_function

def ai_completion(prompt, role="system", temperature=0.85, tokens=450):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": role},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        max_tokens=tokens
    )
    return response["choices"][0]["message"]["content"]

# =========== AI AGENTS ============
class FinanceAgent:
    role = "You are an expert AI financial coach. Help the user optimize personal finances, savings, budgeting, and investing."

    def respond(self, message):
        log_event("FinanceAgent Invoked", {"query": message})
        return ai_completion(message, role=self.role)

class MindAgent:
    role = "You are a compassionate AI mental health coach. Offer support, mindset shifts, and emotional clarity using science-backed methods."

    def respond(self, message):
        log_event("MindAgent Invoked", {"query": message})
        return ai_completion(message, role=self.role)

# Instance Initialization
finance_agent = FinanceAgent()
mind_agent = MindAgent()

# ========== VOICE INTERFACE ==========
def text_to_speech(text, voice="Rachel"):
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice}"
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json"
    }
    payload = {
        "text": text,
        "voice_settings": {"stability": 0.7, "similarity_boost": 0.8}
    }
    response = requests.post(url, headers=headers, json=payload)
    return response.content

# ========== ALGORAND AUTH ==========
def verify_algorand_wallet(wallet_address):
    try:
        headers = {"X-Algo-API-Token": ALGOD_TOKEN}
        response = requests.get(f"{ALGOD_NODE}/v2/accounts/{wallet_address}", headers=headers)
        return response.status_code == 200
    except Exception as e:
        log_event("Algorand Verification Failed", {"error": str(e)})
        return False

# ========== ROUTES ==========

@app.route("/")
def root():
    return jsonify({"status": "Neuronexus AI Engine operational", "agents": ["Finance", "Mind"]})

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
    if not text:
        return error_response("Missing text.")
    audio = text_to_speech(text)
    return audio, 200, {"Content-Type": "audio/mpeg"}

@app.route("/verify", methods=["POST"])
@require_json
def wallet_verify():
    data = request.get_json()
    wallet = data.get("wallet")
    if not wallet:
        return error_response("Missing wallet address.")
    valid = verify_algorand_wallet(wallet)
    return jsonify({"verified": valid})

# ========== MINI ROUTES FOR HACK FUN ==========
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

# ========== BOOTSTRAP ==========
if __name__ == "__main__":
    log_event("Neuronexus Launched")
    app.run(debug=True, port=5050)
# === CONTINUED FROM PART 1 ===

# ========= CREATIVE AGENT (REDDIT / CONTENT) ==========
class CreativeAgent:
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

creative_agent = CreativeAgent()

@app.route("/creative/reddit", methods=["GET"])
def creative_reddit():
    topic = request.args.get("topic", "selfimprovement")
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


# ========= VIDEO COACH PLACEHOLDER (TAVUS SIM) ==========
class VideoCoach:
    def simulate_session(self, prompt):
        reply = ai_completion(f"As a video coach, respond to: '{prompt}' with confident language & interview feedback.")
        return f"🎥 Tavus-style AI says: {reply}\n\n(Note: This is a placeholder. Future integration with Tavus API.)"

video_coach = VideoCoach()

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
USER_SUBSCRIPTIONS = {
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


# ========== LIGHTWEIGHT LOCAL DATASTORE ==========
class DataStore:
    memory = {}

    def remember(self, user_id, key, value):
        if user_id not in self.memory:
            self.memory[user_id] = {}
        self.memory[user_id][key] = value
        log_event("MemoryUpdate", {"user": user_id, "key": key, "value": value})

    def recall(self, user_id, key):
        return self.memory.get(user_id, {}).get(key, None)

datastore = DataStore()

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
# === CONTINUED FROM PART 2 ===

from algosdk.v2client import algod
from algosdk import account, mnemonic
import base64
import string
import secrets

# ========= ALGOSDK CLIENT ==========
algod_client = algod.AlgodClient(ALGOD_TOKEN, ALGOD_NODE)

def get_account_balance(address):
    try:
        acct_info = algod_client.account_info(address)
        return acct_info.get("amount", 0)
    except Exception as e:
        log_event("Algorand Balance Fetch Error", {"error": str(e)})
        return 0

@app.route("/wallet/balance", methods=["POST"])
@require_json
def wallet_balance():
    data = request.get_json()
    wallet = data.get("wallet")
    if not wallet:
        return error_response("Wallet address required.")
    balance = get_account_balance(wallet)
    return jsonify({"wallet": wallet, "balance_microalgos": balance})


# ========= BASIC CONTRACT SIM (STATEFUL TX PLACEHOLDER) ==========
@app.route("/wallet/contract", methods=["POST"])
@require_json
def simulate_contract():
    data = request.get_json()
    wallet = data.get("wallet")
    action = data.get("action")
    if not all([wallet, action]):
        return error_response("Missing data.")
    log_event("SimulatedContractCall", {"wallet": wallet, "action": action})
    return jsonify({"status": "Contract logic accepted (simulation)"})

# ========= USER REGISTRATION / AUTH ==========
users = {}

def generate_token(length=32):
    alphabet = string.ascii_letters + string.digits
    return ''.join(secrets.choice(alphabet) for _ in range(length))

@app.route("/register", methods=["POST"])
@require_json
def register():
    data = request.get_json()
    wallet = data.get("wallet")
    if not wallet:
        return error_response("Wallet required.")
    if wallet in users:
        return jsonify({"status": "Already registered", "token": users[wallet]["token"]})
    token = generate_token()
    users[wallet] = {
        "token": token,
        "created": datetime.datetime.utcnow().isoformat() + "Z"
    }
    log_event("NewUser", {"wallet": wallet})
    return jsonify({"status": "Registered", "token": token})


# ========= ASCII MEME GENERATOR ==========
ASCII_FACES = [
    "(¬‿¬)", "(☞ﾟヮﾟ)☞", "ʕ•ᴥ•ʔ", "(づ｡◕‿‿◕｡)づ", "(╯°□°）╯︵ ┻━┻", "¯\\_(ツ)_/¯", "ಠ_ಠ", "(∩^o^)⊃━☆ﾟ.*･｡ﾟ"
]

@app.route("/ascii-meme", methods=["GET"])
def ascii_meme():
    caption = creative_agent.meme_caption(random.choice(["startups", "AI", "blockchain"]))
    face = random.choice(ASCII_FACES)
    return jsonify({
        "caption": caption,
        "ascii_face": face,
        "display": f"{caption} {face}"
    })


# ========= MULTISTAGE AI CHAIN: CAREER → VOICE → REDDIT ==========
@app.route("/chain/career-pitch", methods=["POST"])
@require_json
def chained_agent_flow():
    data = request.get_json()
    goal = data.get("goal", "get a tech job at a cool startup")

    # 1. Career Agent Drafts Pitch
    pitch_prompt = f"Generate a confident, human-sounding elevator pitch for someone who wants to {goal}."
    pitch_text = ai_completion(pitch_prompt, role="AI career coach.", tokens=180)
    
    # 2. Convert to Audio
    pitch_audio = text_to_speech(pitch_text)

    # 3. Convert into Reddit post suggestion
    reddit_title_prompt = f"Turn this into a motivating Reddit post title: {pitch_text}"
    reddit_title = ai_completion(reddit_title_prompt, role="Clever Reddit content bot.", tokens=50)

    return {
        "pitch_text": pitch_text,
        "reddit_title": reddit_title,
        "audio_mpeg_base64": base64.b64encode(pitch_audio).decode("utf-8")
    }

# === CONTINUED FROM PART 3 ===

# ========== PROMPT REWRITER ==========
class RewriterAgent:
    role = "You are a prompt-optimizer AI that rewrites vague or confusing queries into clear, precise language."

    def refine(self, raw_prompt):
        prompt = f"Rewrite the following prompt for an AI coach: '{raw_prompt}'"
        return ai_completion(prompt, role=self.role, tokens=100)

rewriter_agent = RewriterAgent()


# ========== AGENT ROUTER ==========
class Router:
    def route(self, prompt):
        lowered = prompt.lower()
        if any(word in lowered for word in ["save", "invest", "money", "debt"]):
            return finance_agent
        elif any(word in lowered for word in ["anxious", "sad", "confidence", "stress", "overwhelmed"]):
            return mind_agent
        elif any(word in lowered for word in ["post", "reddit", "caption", "meme", "funny"]):
            return creative_agent
        elif any(word in lowered for word in ["job", "career", "interview", "resume"]):
            return video_coach
        else:
            return finance_agent  # Fallback

router = Router()

@app.route("/smart-coach", methods=["POST"])
@require_json
def smart_coach():
    data = request.get_json()
    prompt = data.get("prompt", "")
    if not prompt:
        return error_response("Missing prompt.")
    cleaned_prompt = rewriter_agent.refine(prompt)
    selected_agent = router.route(cleaned_prompt)
    if isinstance(selected_agent, VideoCoach):
        response = selected_agent.simulate_session(cleaned_prompt)
    else:
        response = selected_agent.respond(cleaned_prompt)
    return jsonify({"reply": response, "routed_by": selected_agent.__class__.__name__})


# ========== ANALYTICS LOGGER ==========
analytics_log = []

@app.route("/analytics/ping", methods=["POST"])
@require_json
def ping():
    data = request.get_json()
    route = data.get("route")
    timestamp = datetime.datetime.utcnow().isoformat()
    analytics_log.append({"route": route, "ts": timestamp})
    log_event("AnalyticsPing", {"route": route})
    return jsonify({"status": "logged"})


# ========== SUPABASE STUB (STRUCTURE ONLY) ==========
# This part simulates what a future Supabase table schema might look like
supabase_model = {
    "users": {
        "wallet": "text",
        "token": "text",
        "created_at": "timestamp",
        "subscription": "enum:free|premium",
        "profile": {
            "goals": "text[]",
            "mood": "text",
            "agent_history": "jsonb"
        }
    }
}

@app.route("/supabase/schema", methods=["GET"])
def show_supabase_schema():
    return jsonify(supabase_model)


# ========== OPTIONAL: POSTPROCESSOR ==========
class PostProcessor:
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

post_processor = PostProcessor()

@app.route("/postprocess", methods=["POST"])
@require_json
def run_postprocess():
    data = request.get_json()
    text = data.get("text", "")
    if not text:
        return error_response("Missing text.")
    score = post_processor.sentiment_score(text)
    tags = post_processor.tag_keywords(text)
    return jsonify({"sentiment": score, "tags": tags})
# === CONTINUED FROM PART 4 ===

# ========= THREAD CONTEXT MANAGER ==========
thread_contexts = {}

def update_thread(user_id, message):
    if user_id not in thread_contexts:
        thread_contexts[user_id] = []
    thread_contexts[user_id].append({
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "message": message
    })
    if len(thread_contexts[user_id]) > 20:
        thread_contexts[user_id] = thread_contexts[user_id][-20:]

def get_thread_summary(user_id):
    messages = thread_contexts.get(user_id, [])
    history = "\n".join([f"{m['timestamp']}: {m['message']}" for m in messages])
    if not history:
        return "No prior context."
    return ai_completion(f"Summarize this chat log: {history}", tokens=100, role="Conversation memory summarizer.")

# ========= TASK QUEUE ENGINE ==========
class Task:
    def __init__(self, name, agent, prompt, postprocess=False, speak=False):
        self.name = name
        self.agent = agent
        self.prompt = prompt
        self.postprocess = postprocess
        self.speak = speak
        self.output = None
        self.tags = []
        self.audio = None

    def run(self):
        if isinstance(self.agent, VideoCoach):
            self.output = self.agent.simulate_session(self.prompt)
        else:
            self.output = self.agent.respond(self.prompt)

        if self.postprocess:
            self.tags = post_processor.tag_keywords(self.output)

        if self.speak:
            self.audio = base64.b64encode(text_to_speech(self.output)).decode("utf-8")
        return self

class Workflow:
    def __init__(self, name):
        self.name = name
        self.tasks = []

    def add_task(self, task: Task):
        self.tasks.append(task)

    def run_all(self):
        results = []
        for task in self.tasks:
            results.append(task.run().__dict__)
        return results

# ========= EXAMPLE WORKFLOWS ==========
def build_job_hunt_workflow(goal="break into product management"):
    wf = Workflow(name="Career Launch")
    wf.add_task(Task("Rewrite Goal", rewriter_agent, f"Clarify this career goal: {goal}"))
    wf.add_task(Task("Interview Prep", video_coach, f"Simulate interview advice for: {goal}", speak=True))
    wf.add_task(Task("Motivational Post", creative_agent, f"Write an r/selfimprovement post for someone trying to {goal}", postprocess=True))
    return wf

def build_mood_reset_workflow(feeling="overwhelmed"):
    wf = Workflow(name="Mood Reset")
    wf.add_task(Task("Validate Emotion", mind_agent, f"I'm feeling {feeling}, what should I do?", speak=True))
    wf.add_task(Task("Suggest Meme", creative_agent, f"Make a relatable meme caption about feeling {feeling}", postprocess=True))
    return wf

@app.route("/workflow/jobhunt", methods=["POST"])
@require_json
def run_jobhunt():
    data = request.get_json()
    goal = data.get("goal", "get a remote developer job")
    user = data.get("user_id", "anon")
    summary = get_thread_summary(user)
    update_thread(user, f"[UserGoal]: {goal}")
    wf = build_job_hunt_workflow(goal)
    results = wf.run_all()
    return jsonify({"workflow": "Career Launch", "steps": results, "context_summary": summary})


@app.route("/workflow/moodreset", methods=["POST"])
@require_json
def run_moodreset():
    data = request.get_json()
    mood = data.get("feeling", "burned out")
    user = data.get("user_id", "anon")
    summary = get_thread_summary(user)
    update_thread(user, f"[Mood]: {mood}")
    wf = build_mood_reset_workflow(mood)
    results = wf.run_all()
    return jsonify({"workflow": "Mood Reset", "steps": results, "context_summary": summary})
# === CONTINUED FROM PART 5 ===

# ========= FRONTEND ROUTE REGISTER ==========
api_registry = {
    "AI Coach Prompt": {
        "endpoint": "/smart-coach",
        "method": "POST",
        "input": ["prompt"],
        "agent": "Router → (Finance, Mind, Creative, Video)"
    },
    "Voice Response": {
        "endpoint": "/speak",
        "method": "POST",
        "input": ["text"],
        "output": "audio/mpeg base64"
    },
    "Generate Reddit Post": {
        "endpoint": "/creative/reddit",
        "method": "GET",
        "input": ["topic (optional)"],
        "agent": "CreativeAgent"
    },
    "Run Job Hunt Workflow": {
        "endpoint": "/workflow/jobhunt",
        "method": "POST",
        "input": ["goal", "user_id"],
        "multi-agent": True
    },
    "Register Wallet": {
        "endpoint": "/register",
        "method": "POST",
        "input": ["wallet"],
        "returns": "Auth token"
    },
    "Check Balance": {
        "endpoint": "/wallet/balance",
        "method": "POST",
        "input": ["wallet"],
        "returns": "MicroAlgos"
    },
    "ASCII Meme Gen": {
        "endpoint": "/ascii-meme",
        "method": "GET",
        "returns": "Meme caption + ASCII face"
    }
}

@app.route("/api/routes", methods=["GET"])
def list_routes():
    return jsonify({"available_routes": api_registry})


# ========= UI PROMPT TEMPLATE SUGGESTIONS ==========
suggestions = {
    "career": [
        "Help me explain my career change from finance to UX design",
        "Simulate a video coach preparing me for a startup PM role"
    ],
    "money": [
        "Make a budget plan using AI — I'm drowning in coffee expenses ☕",
        "Help me automate my savings to reach $5k in 6 months"
    ],
    "mental health": [
        "I feel imposter syndrome — what should I do?",
        "How can I manage burnout while working remotely?"
    ],
    "creative": [
        "Write a sarcastic Reddit post about overcomplicating simple ideas",
        "Suggest a meme caption mocking people who ‘grind’ 25/8"
    ]
}

@app.route("/suggestions", methods=["GET"])
def get_ui_suggestions():
    return jsonify(suggestions)
# === Structured Goal Schema Definition ===

example_goal_dsl = {
    "goal": "get hired as a UX designer at a startup",
    "phases": [
        {"name": "clarifyGoal", "agent": "RewriterAgent"},
        {"name": "preparePitch", "agent": "VideoCoach"},
        {"name": "createRedditPost", "agent": "CreativeAgent"},
        {"name": "budgetForGapMonths", "agent": "FinanceAgent"},
        {"name": "motivate", "agent": "MindAgent"}
    ]
}

def compile_dsl(dsl_json):
    wf = Workflow(name=dsl_json["goal"])
    for phase in dsl_json["phases"]:
        agent = globals()[phase["agent"]]()
        wf.add_task(Task(phase["name"], agent, f"Phase '{phase['name']}' for goal: {dsl_json['goal']}", speak=False))
    return wf

@app.route("/workflow/custom", methods=["POST"])
@require_json
def custom_workflow():
    data = request.get_json()
    wf = compile_dsl(data)
    return jsonify({"compiled_for": wf.name, "steps": [t.name for t in wf.tasks]})
FROM python:3.11-slim
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
EXPOSE 5050
CMD ["python", "app.py"]
[build]
  functions = "functions"
  command = "python app.py"
  publish = "public"

[[redirects]]
  from = "/*"
  to = "/index.html"
  status = 200
@app.route("/webhook/<source>", methods=["POST"])
def webhook(source):
    payload = request.json
    log_event(f"Webhook received from {source}", {"payload": payload})
    return jsonify({"status": "ok", "source": source})
def explain_agent_chain(goal):
    prompt = f"""
    Break down the goal: '{goal}' into logical steps and assign each to:
    - FinanceAgent
    - MindAgent
    - CreativeAgent
    - RewriterAgent
    - VideoCoach
    """
    breakdown = ai_completion(prompt, role="AI workflow architect", tokens=300)
    return breakdown

@app.route("/goals/auto-assign", methods=["POST"])
@require_json
def goal_auto_assign():
    data = request.get_json()
    goal = data.get("goal")
    mapping = explain_agent_chain(goal)
    return jsonify({"assignments": mapping})
@app.route("/agent/debug/<agent_name>", methods=["POST"])
@require_json
def debug_agent(agent_name):
    data = request.get_json()
    test_prompt = data.get("prompt", "Hello")
    try:
        agent_cls = globals()[agent_name]
        instance = agent_cls()
        reply = instance.respond(test_prompt)
        return jsonify({"reply": reply, "agent": agent_name})
    except Exception as e:
        return error_response(f"Agent failed: {str(e)}")
import React from 'react';
import { SafeAreaView, View, TextInput, Button, Text, ScrollView, Alert } from 'react-native';

export default function App() {
  const [prompt, setPrompt] = React.useState('');
  const [response, setResponse] = React.useState('');

  const askNeuronexus = async () => {
    try {
      const res = await fetch('https://your-deploy-url/smart-coach', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt })
      });
      const data = await res.json();
      setResponse(data.reply);
    } catch (err) {
      Alert.alert('Error', 'Could not reach Neuronexus 😢');
    }
  };

  return (
    <SafeAreaView style={{ padding: 20 }}>
      <Text style={{ fontSize: 24, fontWeight: 'bold' }}>Neuronexus Mobile</Text>
      <TextInput
        placeholder="Ask me anything..."
        onChangeText={setPrompt}
        value={prompt}
        style={{ borderWidth: 1, padding: 10, marginVertical: 10 }}
      />
      <Button title="Ask" onPress={askNeuronexus} />
      <ScrollView style={{ marginTop: 20 }}>
        <Text>{response}</Text>
      </ScrollView>
    </SafeAreaView>
  );
}
from tinydb import TinyDB, Query
db = TinyDB("nexus_memory.json")

@app.route("/memory/persist", methods=["POST"])
@require_json
def persist_memory():
    data = request.get_json()
    user = data.get("user")
    key = data.get("key")
    value = data.get("value")
    db.insert({"user": user, "key": key, "value": value, "ts": datetime.datetime.utcnow().isoformat()})
    return jsonify({"status": "stored"})

@app.route("/memory/retrieve", methods=["POST"])
@require_json
def retrieve_memory():
    data = request.get_json()
    user = data.get("user")
    key = data.get("key")
    Memory = Query()
    result = db.search((Memory.user == user) & (Memory.key == key))
    return jsonify({"values": result})
@app.route("/analytics/view", methods=["GET"])
def view_analytics():
    summary = {}
    for ping in analytics_log:
        route = ping["route"]
        summary[route] = summary.get(route, 0) + 1
    return jsonify({"usage": summary})
@app.route("/session/sync", methods=["POST"])
@require_json
def session_sync():
    data = request.get_json()
    wallet = data.get("wallet")
    token = users.get(wallet, {}).get("token", None)
    return jsonify({"wallet": wallet, "token": token})
@app.route("/knowledge/ingest", methods=["POST"])
@require_json
def ingest_doc():
    data = request.get_json()
    content = data.get("text")
    summary = ai_completion(f"Summarize this user knowledge doc:\n\n{content}", tokens=150)
    datastore.remember("GLOBAL", "knowledge", summary)
    return jsonify({"summary": summary})
plugin_registry = {}

def register_agent(agent_name, agent_obj):
    plugin_registry[agent_name] = agent_obj

@app.route("/agent/register", methods=["POST"])
@require_json
def dynamic_agent_register():
    data = request.get_json()
    name = data.get("name")
    prompt = data.get("prompt")
    class CustomPlugin:
        def respond(self, query):
            return ai_completion(query, role=prompt)
    register_agent(name, CustomPlugin())
    return jsonify({"status": f"{name} registered"})

@app.route("/agent/run/<agent_name>", methods=["POST"])
@require_json
def run_registered_agent(agent_name):
    prompt = request.get_json().get("prompt", "")
    agent = plugin_registry.get(agent_name)
    if not agent:
        return error_response("No such agent.")
    response = agent.respond(prompt)
    return jsonify({"reply": response})
agent_market = {
    "FitnessCoach": {
        "description": "Create custom workout routines & motivation messages",
        "author": "external-dev-42",
        "price_usd": 3.99
    },
    "AI Poet": {
        "description": "Composes emotional poetry and symbolic verse",
        "author": "haikuDAO.eth",
        "price_usd": 2.00
    }
}

@app.route("/marketplace/agents", methods=["GET"])
def list_market_agents():
    return jsonify(agent_market)
badges = {
    "Deep Listener": "Completed 10+ mental health chats",
    "Budget Boss": "Completed 5 custom finance plans",
    "Reddit Rizzler": "Generated 7+ community posts"
}

@app.route("/user/badges", methods=["POST"])
@require_json
def badge_check():
    wallet = request.get_json().get("wallet")
    # Simulate logic
    return jsonify({"wallet": wallet, "earned": ["Deep Listener", "Reddit Rizzler"]})
thread_live = {}

@app.route("/thread/live", methods=["POST"])
@require_json
def thread_live_prompt():
    data = request.get_json()
    user = data.get("user", "anon")
    prompt = data.get("prompt")
    context = get_thread_summary(user)
    reply = ai_completion(f"With context:\n{context}\nRespond to: {prompt}")
    update_thread(user, prompt)
    return jsonify({"context": context, "reply": reply})
startup_templates = {
    "founder_coaching": "You are an AI startup mentor. Help with pitching, fundraising, hiring, and mental balance.",
    "artist_block": "You are a creative unblocker. Prompt the user through curiosity and imagination exercises.",
    "resume_magic": "You are a resume whisperer. Rewrite dull bullet points like you’re recruiting for NASA."
}

@app.route("/prompt/template", methods=["GET"])
def show_templates():
    return jsonify(startup_templates)
@app.route("/auto/mission", methods=["POST"])
@require_json
def propose_mission():
    data = request.get_json()
    mood = data.get("mood", "undefined")
    thread = get_thread_summary(data.get("user", "anon"))
    prompt = f"""
    Based on this user mood: '{mood}' and recent conversation: '{thread}',
    propose a 3-step mission for personal growth with agent assignments.
    """
    plan = ai_completion(prompt, tokens=250, role="AI mission composer")
    return jsonify({"proposed_mission": plan})
@app.route("/timeline/create", methods=["POST"])
@require_json
def timeline_create():
    data = request.get_json()
    steps = data.get("steps", [])
    timeline = []
    now = datetime.datetime.utcnow()
    for i, s in enumerate(steps):
        timeline.append({
            "time": (now + datetime.timedelta(hours=i)).isoformat(),
            "task": s.get("task"),
            "agent": s.get("agent")
        })
    return jsonify({"timeline": timeline})
agent_logs = []

@app.route("/agent/reflect", methods=["POST"])
@require_json
def reflect_agent():
    data = request.get_json()
    agent = data.get("agent")
    prompt = data.get("prompt")
    reply = data.get("reply")
    reflection = ai_completion(
        f"As {agent}, reflect on your response to '{prompt}': '{reply}'. Was it helpful?",
        role="Agent Reflection Coach"
    )
    agent_logs.append({
        "agent": agent,
        "prompt": prompt,
        "reflection": reflection
    })
    return jsonify({"reflection": reflection})
@app.route("/system/pulse", methods=["GET"])
def pulse():
    uptime = datetime.datetime.utcnow() - datetime.datetime.fromisoformat(start_time)
    active_agents = list(plugin_registry.keys()) + ["FinanceAgent", "MindAgent", "CreativeAgent", "VideoCoach"]
    return jsonify({
        "status": "🔥 LIVE",
        "uptime": str(uptime),
        "active_agents": active_agents,
        "memory_entries": len(thread_contexts),
        "custom_plugins": len(plugin_registry)
    })
import datetime # Make sure datetime is imported if you use it within the class
# (Assuming DataStore, Router, RewriterAgent, PostProcessor, Task, Workflow, and your specific agents
# like FinanceAgent, MindAgent, CreativeAgent, VideoCoach are defined and available globally or passed in)

class SentientChainExecutor:
    """
    Orchestrates multi-agent interactions, leveraging memory, agent routing,
    and prompt optimization for complex user queries.
    """
    def __init__(self, memory_store, agent_router, rewriter_agent, post_processor, thread_contexts):
        """
        Initializes the SentientChainExecutor with necessary components.

        Args:
            memory_store (DataStore): An instance of the local data store for memory recall/save.
            agent_router (Router): An instance of the Router to select appropriate agents.
            rewriter_agent (RewriterAgent): An instance of the RewriterAgent for prompt optimization.
            post_processor (PostProcessor): An instance of the PostProcessor for output analysis.
            thread_contexts (dict): A mutable dictionary to store live conversation contexts.
        """
        self.memory = memory_store
        self.router = agent_router
        self.rewriter = rewriter_agent
        self.post_processor = post_processor
        self.thread_contexts = thread_contexts # Pass the global thread_contexts dict

    def _get_thread_summary(self, user_id):
        """Internal helper to get summarized thread context."""
        messages = self.thread_contexts.get(user_id, [])
        history = "\n".join([f"{m['timestamp']}: {m['message']}" for m in messages])
        if not history:
            return "No prior context."
        # Assuming ai_completion is globally available or passed in
        return ai_completion(f"Summarize this chat log: {history}", tokens=100, role="Conversation memory summarizer.")

    def _update_thread(self, user_id, message):
        """Internal helper to update the live thread context."""
        if user_id not in self.thread_contexts:
            self.thread_contexts[user_id] = []
        self.thread_contexts[user_id].append({
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "message": message
        })
        if len(self.thread_contexts[user_id]) > 20: # Keep thread concise
            self.thread_contexts[user_id] = self.thread_contexts[user_id][-20:]

    def execute_chain(self, user_id, raw_prompt, persist_memory=False, speak_output=False):
        """
        Executes a multi-stage AI chain based on a user prompt, utilizing various agents.

        Args:
            user_id (str): Identifier for the user, for memory and thread context.
            raw_prompt (str): The initial user prompt.
            persist_memory (bool): Whether to save the interaction to persistent memory.
            speak_output (bool): Whether to convert the final response to speech.

        Returns:
            dict: A dictionary containing the processed output, agent used, tags, sentiment, and audio (if requested).
        """
        log_event("Chain Execution Initiated", {"user_id": user_id, "prompt": raw_prompt})
        self._update_thread(user_id, f"User: {raw_prompt}")

        # 1. Get current context summary
        context_summary = self._get_thread_summary(user_id)
        logging.info(f"Context for {user_id}: {context_summary}")

        # 2. Refine the prompt using RewriterAgent
        refined_prompt = self.rewriter.refine(raw_prompt)
        logging.info(f"Refined prompt: {refined_prompt}")

        # 3. Route to the appropriate agent
        # Combine context with the prompt for better routing/response if desired,
        # but for simplicity here, just the refined_prompt for routing.
        selected_agent = self.router.route(refined_prompt)
        agent_name = selected_agent.__class__.__name__
        logging.info(f"Routed to agent: {agent_name}")

        # 4. Get response from the selected agent
        agent_response = ""
        if isinstance(selected_agent, VideoCoach):
            # VideoCoach.simulate_session doesn't take 'role'
            agent_response = selected_agent.simulate_session(refined_prompt)
        else:
            # Other agents have a 'respond' method that takes prompt
            agent_response = selected_agent.respond(refined_prompt)

        self._update_thread(user_id, f"Agent ({agent_name}): {agent_response}")
        log_event("Agent Response Generated", {"agent": agent_name, "response_len": len(agent_response)})

        # 5. Post-process the agent's response
        sentiment = self.post_processor.sentiment_score(agent_response)
        tags = self.post_processor.tag_keywords(agent_response)
        logging.info(f"Post-processed: Sentiment={sentiment}, Tags={tags}")

        # 6. Optionally save to persistent memory
        if persist_memory:
            self.memory.remember(user_id, "last_interaction", {
                "prompt": raw_prompt,
                "response": agent_response,
                "agent": agent_name,
                "timestamp": datetime.datetime.utcnow().isoformat()
            })
            log_event("Interaction Saved to Persistent Memory", {"user_id": user_id})

        # 7. Optionally convert to speech
        audio_base64 = None
        if speak_output:
            try:
                audio_content = text_to_speech(agent_response) # text_to_speech is a global function
                audio_base64 = base64.b64encode(audio_content).decode("utf-8")
                log_event("Text-to-Speech Generated")
            except Exception as e:
                logging.error(f"Failed to generate speech: {e}")
                audio_base64 = None # Ensure it's None on failure

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
# ========== FINAL BOOTSTRAP ==========

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Start Neuronexus AI OS")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5050)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    log_event("Neuronexus system booted.", {"host": args.host, "port": args.port})
    app.run(host=args.host, port=args.port, debug=args.debug)
