import os
import openai
import requests
import hashlib
import json
import logging
import random
import datetime
import base64
import secrets
import string
import re # For regular expressions in tool extraction

from flask import Flask, request, jsonify
from functools import wraps

# Imports for Algorand SDK
from algosdk.v2client import algod

# Imports for TinyDB
from tinydb import TinyDB, Query

# ==== INITIAL SETUP ====
app = Flask(__name__)

# === APPLICATION START TIME ===
start_time = datetime.datetime.utcnow()

# === ENVIRONMENT CONFIG ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ALGOD_NODE = os.getenv("ALGOD_NODE", 'https://testnet-api.algonode.cloud')
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
            return error_response("Request must be JSON", 415)
        return f(*args, **kwargs)
    return decorated_function

def ai_completion(prompt, role="system", temperature=0.85, tokens=450, json_mode=False):
    """
    Calls the OpenAI Chat Completion API.
    Includes basic error handling for API calls.
    Added json_mode for structured outputs.
    """
    if not openai.api_key:
        log_event("AI Completion Error", {"reason": "OpenAI API key not set"})
        raise ValueError("OpenAI API key is not configured.")
    try:
        messages = [
            {"role": "system", "content": role},
            {"role": "user", "content": prompt}
        ]
        
        # Determine response format based on json_mode
        response_format = {"type": "json_object"} if json_mode else {"type": "text"}

        response = openai.ChatCompletion.create(
            model="gpt-4", # Can change to gpt-3.5-turbo if gpt-4 access is limited
            messages=messages,
            temperature=temperature,
            max_tokens=tokens,
            response_format=response_format # Use the new response_format parameter
        )
        content = response["choices"][0]["message"]["content"]
        if json_mode:
            try:
                return json.loads(content)
            except json.JSONDecodeError as e:
                log_event("AI Completion JSON Decode Error", {"error": str(e), "content": content})
                raise RuntimeError(f"Failed to decode JSON from AI response: {e}")
        return content
    except openai.error.OpenAIError as e:
        log_event("OpenAI API Error", {"error": str(e), "prompt_len": len(prompt)})
        raise RuntimeError(f"OpenAI API call failed: {e}")
    except Exception as e:
        log_event("AI Completion Unexpected Error", {"error": str(e), "prompt_len": len(prompt)})
        raise RuntimeError(f"An unexpected error occurred during AI completion: {e}")

# =========== TOOLS AND KNOWLEDGE BASES ============
class Tool:
    """Base class for all tools."""
    def __init__(self, name, description):
        self.name = name
        self.description = description

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("Each tool must implement the __call__ method.")

class KnowledgeBaseTool(Tool):
    """External knowledge base for agents to query."""
    def __init__(self):
        super().__init__("KnowledgeBase", "Provides current stock prices and market news summaries. Call with 'get_stock_data(TICKER)' or 'get_market_news_summary()'.")
        self.simulated_data = {
            "AAPL": {"price": 175.25, "change": "+1.10", "volume": "75M"},
            "GOOG": {"price": 180.50, "change": "-0.55", "volume": "40M"},
            "MSFT": {"price": 440.10, "change": "+2.00", "volume": "60M"},
            "AMZN": {"price": 190.70, "change": "-0.90", "volume": "90M"}
        }

    def get_stock_data(self, ticker):
        """Simulates fetching real-time stock data for a given ticker."""
        return self.simulated_data.get(ticker.upper(), {"error": "Ticker not found or data unavailable."})

    def get_market_news_summary(self):
        """Simulates fetching a brief market news summary."""
        return "The market is showing mixed signals today. Tech stocks are slightly up, while consumer discretionary is pulling back. Inflation concerns persist, but interest rate cut hopes provide some optimism."

    def __call__(self, method_name, *args):
        """Allows calling specific methods of the tool."""
        if hasattr(self, method_name) and callable(getattr(self, method_name)):
            method = getattr(self, method_name)
            # Basic argument handling, could be more robust
            if method_name == "get_stock_data" and args:
                return method(args[0])
            elif method_name == "get_market_news_summary":
                return method()
        return {"error": f"Method {method_name} not found or unsupported by KnowledgeBaseTool."}

class CalculatorTool(Tool):
    """Performs basic arithmetic calculations."""
    def __init__(self):
        super().__init__("Calculator", "Performs arithmetic operations. Call with 'calculate(expression)' e.g., 'calculate(2 + 2 * 3)'. Supports +, -, *, /.")

    def calculate(self, expression):
        """Safely evaluates a mathematical expression."""
        try:
            # Basic security: only allow digits, operators, and parentheses
            if not re.match(r"^[0-9+\-*/(). ]+$", expression):
                return {"error": "Invalid characters in expression."}
            return {"result": eval(expression)} # eval is dangerous, but for sim, okay
        except Exception as e:
            return {"error": f"Calculation failed: {str(e)}"}

    def __call__(self, method_name, *args):
        if method_name == "calculate" and args:
            return self.calculate(args[0])
        return {"error": f"Method {method_name} not found or unsupported by CalculatorTool."}

class CalendarTool(Tool):
    """Manages calendar events."""
    def __init__(self):
        super().__init__("Calendar", "Adds events to a calendar. Call with 'add_event(title, date, time)' e.g., 'add_event(\"Meeting\", \"2025-07-10\", \"10:00\")'.")
        self.events = [] # In-memory storage for simplicity

    def add_event(self, title, date_str, time_str):
        """Adds a new event to the calendar."""
        try:
            event_datetime = datetime.datetime.fromisoformat(f"{date_str}T{time_str}")
            event = {"title": title, "datetime": event_datetime.isoformat()}
            self.events.append(event)
            return {"status": "Event added successfully", "event": event}
        except ValueError:
            return {"error": "Invalid date or time format. Use YYYY-MM-DD and HH:MM."}

    def __call__(self, method_name, *args):
        if method_name == "add_event" and len(args) == 3:
            return self.add_event(args[0], args[1], args[2])
        return {"error": f"Method {method_name} not found or unsupported by CalendarTool."}

class WebSearchTool(Tool):
    """Simulates web search for general information."""
    def __init__(self):
        super().__init__("WebSearch", "Performs a simulated web search. Call with 'search(query)'.")

    def search(self, query):
        """Simulates a web search result."""
        # In a real application, this would use a proper search API
        simulated_results = {
            "latest AI trends": "AI trends indicate a rise in multimodal models, explainable AI, and edge computing.",
            "benefits of meditation": "Meditation can reduce stress, improve focus, and enhance emotional regulation.",
            "history of python": "Python was created by Guido van Rossum and first released in 1991."
        }
        # Simple keyword matching for demo
        for key, value in simulated_results.items():
            if query.lower() in key.lower():
                return {"result": value}
        return {"result": f"No specific web result found for '{query}'. This is a simulated search."}

    def __call__(self, method_name, *args):
        if method_name == "search" and args:
            return self.search(args[0])
        return {"error": f"Method {method_name} not found or unsupported by WebSearchTool."}


# Tool Registry: A dictionary of callable tool instances
TOOL_REGISTRY = {
    "KnowledgeBase": KnowledgeBaseTool(),
    "Calculator": CalculatorTool(),
    "Calendar": CalendarTool(),
    "WebSearch": WebSearchTool()
}


# =========== AI AGENTS ============
class FinanceAgent:
    """An AI financial coach."""
    role = "You are an expert AI financial coach. Help the user optimize personal finances, savings, budgeting, and investing."

    def respond(self, message):
        log_event("FinanceAgent Invoked", {"query": message})
        # Finance agent can still respond directly, but complex queries
        # might be handled by PersonalAssistantAgent first.
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
    def simulate_session(self, prompt, tokens=450):
        reply = ai_completion(f"As a video coach, respond to: '{prompt}' with confident language & interview feedback.", tokens=tokens)
        return f"🎥 Tavus-style AI says: {reply}\n\n(Note: This is a placeholder. Future integration with Tavus API.)"

class RewriterAgent:
    """An AI prompt optimizer."""
    role = "You are a prompt-optimizer AI that rewrites vague or confusing queries into clear, precise language."

    def refine(self, raw_prompt, context=""):
        full_prompt = f"Rewrite the following prompt for an AI coach: '{raw_prompt}'\n\nContext for refinement: {context}"
        return ai_completion(full_prompt, role=self.role, tokens=100)

class PostProcessor:
    """Performs post-processing on AI generated text."""
    def tag_keywords(self, text):
        keywords = []
        for word in ["debt", "startup", "job", "anxiety", "invest", "Reddit", "meme", "GPT", "stock", "market", "event", "calendar", "calculate", "search", "web"]:
            if word.lower() in text.lower():
                keywords.append(word)
        return keywords

    def sentiment_score(self, text):
        prompt = f"What is the sentiment of this text? '{text}'. Answer with one word: Positive, Neutral, or Negative."
        result = ai_completion(prompt, role="Sentiment evaluator.", tokens=5)
        return result.strip()

# New: Personal Assistant Agent (The Orchestrator)
class PersonalAssistantAgent:
    """
    An AI personal assistant that can plan and execute tasks using other agents and tools.
    This is the core of the 'more powerful AI component'.
    """
    role = """You are a highly capable AI personal assistant, designed to assist users with a wide range of tasks.
    You can use the following agents and tools to fulfill requests:

    Agents:
    - FinanceAgent: For financial advice, budgeting, investing, and market insights.
    - MindAgent: For mental health support, mindset, and emotional clarity.
    - CreativeAgent: For generating creative content like Reddit posts, memes, and blog ideas.
    - VideoCoach: For job, career, interview, and pitch coaching simulations.

    Tools:
    - KnowledgeBase: Access current stock prices and market news summaries. Methods: get_stock_data(TICKER), get_market_news_summary().
    - Calculator: Perform arithmetic calculations. Method: calculate(EXPRESSION).
    - Calendar: Add events to a calendar. Method: add_event(TITLE, DATE, TIME).
    - WebSearch: Perform a general web search. Method: search(QUERY).

    When a user provides a complex request that requires multiple steps, external data, or different domains,
    you must create a plan in JSON format. The plan should be an array of objects, where each object
    represents a step. Each step must have an 'action' (the tool or agent to use) and 'args' (arguments for the action).
    If an action returns data that needs to be used in a subsequent step, describe how it will be used.

    Example JSON Plan for "What's the current price of Apple stock and how does it compare to Google?"
    ```json
    [
      {"step": 1, "action": "KnowledgeBase", "method": "get_stock_data", "args": ["AAPL"], "description": "Get Apple stock price."},
      {"step": 2, "action": "KnowledgeBase", "method": "get_stock_data", "args": ["GOOG"], "description": "Get Google stock price."},
      {"step": 3, "action": "Calculator", "method": "calculate", "args": ["result of step 1 price - result of step 2 price"], "description": "Compare prices."},
      {"step": 4, "action": "final_response", "args": ["Synthesize information from step 1, 2, and 3 to answer the user's question."], "description": "Provide the final answer."}
    ]
    ```
    Example JSON Plan for "Add a meeting to my calendar for next Monday at 3 PM titled 'Project Alpha Review' and then tell me some tips for improving focus."
    ```json
    [
      {"step": 1, "action": "Calendar", "method": "add_event", "args": ["Project Alpha Review", "2025-07-07", "15:00"], "description": "Add event to calendar (assuming today is 2025-07-01 and next Monday is July 7th)."},
      {"step": 2, "action": "MindAgent", "method": "respond", "args": ["Provide tips for improving focus."], "description": "Get focus tips from MindAgent."},
      {"step": 3, "action": "final_response", "args": ["Confirm event added and provide focus tips."], "description": "Combine responses."}
    ]
    ```

    If a request is simple and directly falls under one agent's domain without needing external tools or multi-step logic, you can respond directly with that agent's output without a JSON plan.
    You MUST output a JSON plan if multiple steps or tools are required.
    """

    def plan_and_execute(self, user_id, raw_prompt, combined_context, available_agents, available_tools):
        """
        The core orchestration method. Generates a plan and executes it.
        """
        log_event("PersonalAssistantAgent: Planning", {"user_id": user_id, "prompt": raw_prompt})

        # Dynamically create the prompt for the planning stage, including available tools/agents
        tool_descriptions = "\n".join([f"- {name}: {tool.description}" for name, tool in available_tools.items()])
        agent_list = ", ".join([agent.__class__.__name__ for agent in available_agents])

        planning_prompt = f"""
        Given the user's request, the conversation context, and the available agents and tools,
        formulate a step-by-step plan in JSON format. Each step should specify the 'action' (tool or agent name),
        the 'method' to call (if a tool), and 'args' for that method.
        If a step's output is needed for a subsequent step, indicate it in the 'args' or 'description'.
        The final step should always be 'final_response' with 'args' being the synthesized answer.

        Available Agents: {agent_list}
        Available Tools:
        {tool_descriptions}

        Conversation Context:
        {combined_context}

        User's Request: "{raw_prompt}"

        Your JSON Plan:
        """
        try:
            # Use JSON mode for the plan generation
            plan = ai_completion(planning_prompt, role=self.role, tokens=500, json_mode=True)
            log_event("PersonalAssistantAgent: Generated Plan", {"user_id": user_id, "plan": plan})
            
            # Execute the plan
            execution_results = {}
            final_output_parts = []

            for step in plan:
                step_number = step.get("step", "N/A")
                action = step.get("action")
                method = step.get("method")
                args = step.get("args", [])
                description = step.get("description", "")

                log_event(f"PersonalAssistantAgent: Executing Step {step_number}", {"action": action, "method": method, "args": args})

                try:
                    # Resolve arguments that reference previous step results
                    resolved_args = []
                    for arg in args:
                        if isinstance(arg, str) and "result of step" in arg:
                            match = re.search(r"result of step (\d+)", arg)
                            if match:
                                prev_step_num = int(match.group(1))
                                if prev_step_num in execution_results and "result" in execution_results[prev_step_num]:
                                    resolved_args.append(execution_results[prev_step_num]["result"])
                                else:
                                    resolved_args.append(f"ERROR: Result for step {prev_step_num} not found.")
                            else:
                                resolved_args.append(arg) # No match, keep original arg
                        else:
                            resolved_args.append(arg)

                    if action == "final_response":
                        final_output_parts.append(resolved_args[0] if resolved_args else description)
                        execution_results[step_number] = {"result": final_output_parts[-1]}
                        break # End of plan

                    elif action in available_tools:
                        tool_instance = available_tools[action]
                        tool_result = tool_instance(method, *resolved_args)
                        execution_results[step_number] = {"action": action, "method": method, "args": resolved_args, "result": tool_result}
                        final_output_parts.append(f"Tool {action}.{method} output: {json.dumps(tool_result)}")

                    elif action in [agent.__class__.__name__ for agent in available_agents]:
                        # Find the agent instance by name
                        agent_instance = next((a for a in available_agents if a.__class__.__name__ == action), None)
                        if agent_instance:
                            # Agent's respond method now takes context-augmented prompt
                            # For tool-calling within orchestrator, simplify direct call
                            if hasattr(agent_instance, method): # e.g., MindAgent.respond, CreativeAgent.reddit_post
                                agent_method = getattr(agent_instance, method)
                                # For agents, the first argument is typically the prompt
                                agent_output = agent_method(resolved_args[0] if resolved_args else description)
                            else:
                                agent_output = f"Error: Agent method '{method}' not found for '{action}'."
                            
                            execution_results[step_number] = {"action": action, "method": method, "args": resolved_args, "result": agent_output}
                            final_output_parts.append(f"Agent {action} output: {agent_output}")
                        else:
                            log_event("PersonalAssistantAgent: Agent Not Found", {"action": action})
                            execution_results[step_number] = {"error": f"Agent '{action}' not found."}
                            final_output_parts.append(f"Error: Agent '{action}' not found.")

                    else:
                        log_event("PersonalAssistantAgent: Unknown Action", {"action": action})
                        execution_results[step_number] = {"error": f"Unknown action: {action}"}
                        final_output_parts.append(f"Error: Unknown action '{action}'.")

                except Exception as step_e:
                    log_event(f"PersonalAssistantAgent: Step {step_number} Failed", {"error": str(step_e), "step_details": step})
                    execution_results[step_number] = {"error": f"Failed to execute step: {str(step_e)}"}
                    final_output_parts.append(f"Error in step {step_number}: {str(step_e)}")
            
            # Final synthesis if no explicit final_response step was found, or if errors occurred
            if not final_output_parts or "Error" in final_output_parts[-1]:
                final_response_text = "I encountered an issue or could not fully complete your request. Here's what I found:\n" + "\n".join([str(res.get("result", res.get("error", ""))) for res in execution_results.values() if res])
                if not final_response_text:
                    final_response_text = "I could not generate a response for your request."
            else:
                final_response_text = final_output_parts[-1] # The last final_response message

            return final_response_text, execution_results

        except RuntimeError as e:
            log_event("PersonalAssistantAgent: Planning Failed", {"user_id": user_id, "error": str(e)})
            return f"I apologize, I couldn't create a plan to fulfill your request: {e}", {}
        except Exception as e:
            log_event("PersonalAssistantAgent: Unexpected Error During Planning/Execution", {"user_id": user_id, "error": str(e)})
            return f"An unexpected error occurred in the assistant: {e}", {}


# Instance Initialization of Agents
finance_agent = FinanceAgent()
mind_agent = MindAgent()
creative_agent = CreativeAgent()
video_coach = VideoCoach()
rewriter_agent = RewriterAgent()
post_processor = PostProcessor()
personal_assistant_agent = PersonalAssistantAgent() # Instantiate the new orchestrator

# A list of agents that the PersonalAssistantAgent can orchestrate
ORCHESTRATED_AGENTS = [
    finance_agent,
    mind_agent,
    creative_agent,
    video_coach # Note: VideoCoach doesn't have a 'respond' method, orchestrator calls simulate_session directly
]


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
        voice_id = "21m00Tzpb8CxoXLnyC0v" # Default to Rachel
        if voice == "Adam":
            voice_id = "pNInz6obpgDQGxUGXP57"
        elif voice == "Rachel":
            voice_id = "21m00Tzpb8CxoXLnyC0v"

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
        response.raise_for_status()
        return response.content
    except requests.exceptions.RequestException as e:
        log_event("ElevenLabs API Error", {"error": str(e), "text_len": len(text)})
        raise RuntimeError(f"ElevenLabs API call failed: {e}")
    except Exception as e:
        log_event("Text-to-Speech Unexpected Error", {"error": str(e), "text_len": len(text)})
        raise RuntimeError(f"An unexpected error occurred during text-to-speech: {e}")

# ========== ALGORAND AUTH ==========
algod_client = algod.AlgodClient(ALGOD_TOKEN, ALGOD_NODE)

def verify_algorand_wallet(wallet_address):
    """
    Verifies if an Algorand wallet address is valid and exists on the network.
    """
    try:
        headers = {}
        if ALGOD_TOKEN:
             headers = {"X-Algo-API-Token": ALGOD_TOKEN}

        response = requests.get(f"{ALGOD_NODE}/v2/accounts/{wallet_address}", headers=headers)
        response.raise_for_status()
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
        acct_info = algod_client.account_info(address)
        return acct_info.get("amount", 0)
    except Exception as e:
        log_event("Algorand Balance Fetch Error", {"error": str(e)})
        return 0

# ========== LIGHTWEIGHT LOCAL DATASTORE (In-memory) ==========
class DataStore:
    """A simple in-memory key-value store for user data."""
    memory = {}

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
db_persistent = TinyDB("nexus_memory.json")

# ========== USER REGISTRATION / AUTH (In-memory) ==========
users = {}

def generate_token(length=32):
    """Generates a random alphanumeric token."""
    alphabet = string.ascii_letters + string.digits
    return ''.join(secrets.choice(alphabet) for _ in range(length))

# ========= THREAD CONTEXT MANAGER (In-memory) ==========
thread_contexts = {}

def update_thread(user_id, message):
    """Updates the conversation thread for a given user."""
    if user_id not in thread_contexts:
        thread_contexts[user_id] = []
    thread_contexts[user_id].append({
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "message": message
    })
    if len(thread_contexts[user_id]) > 20:
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
    """
    Represents a single task in a workflow.
    Simplified as the PersonalAssistantAgent handles the complex orchestration.
    """
    def __init__(self, name, agent, prompt, postprocess=False, speak=False, voice="Rachel"):
        self.name = name
        self.agent = agent
        self.prompt = prompt
        self.postprocess = postprocess
        self.speak = speak
        self.voice = voice
        self.output = None
        self.tags = []
        self.sentiment = None
        self.audio = None

    def run(self):
        """Executes the task using its assigned agent."""
        log_event("Task Running", {"task_name": self.name, "agent": self.agent.__class__.__name__})
        try:
            # The Task class here is simplified as the PersonalAssistantAgent will
            # be responsible for calling agent methods or tools directly now.
            # This 'run' method might only be used for direct single-agent calls
            # that bypass the orchestrator, or for legacy routes.
            if isinstance(self.agent, VideoCoach):
                self.output = self.agent.simulate_session(self.prompt)
            elif hasattr(self.agent, 'respond'):
                self.output = self.agent.respond(self.prompt)
            elif hasattr(self.agent, 'reddit_post') and "reddit" in self.name.lower():
                self.output = self.agent.reddit_post(self.prompt)
            elif hasattr(self.agent, 'meme_caption') and "meme" in self.name.lower():
                self.output = self.agent.meme_caption(self.prompt)
            elif hasattr(self.agent, 'blog_idea') and "blog" in self.name.lower():
                self.output = self.agent.blog_idea(self.prompt)
            elif hasattr(self.agent, 'refine') and "rewrite" in self.name.lower():
                self.output = self.agent.refine(self.prompt)
            else:
                self.output = ai_completion(self.prompt, role=getattr(self.agent, 'role', 'assistant'))

            if self.postprocess:
                self.tags = post_processor.tag_keywords(self.output)
                self.sentiment = post_processor.sentiment_score(self.output)

            if self.speak:
                audio_content = text_to_speech(self.output, voice=self.voice)
                self.audio = base64.b64encode(audio_content).decode("utf-8")
        except Exception as e:
            log_event("Task Failed", {"task_name": self.name, "error": str(e)})
            self.output = f"Error executing task '{self.name}': {str(e)}"
            self.audio = None
        return self

class Workflow:
    """Represents a sequence of tasks to be executed (less relevant with Orchestrator)."""
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

# ========== SENTIENT CHAIN EXECUTOR (Multi-Agent Orchestrator) ==========
class SentientChainExecutor:
    """
    Orchestrates multi-agent interactions, leveraging memory, agent routing,
    and prompt optimization for complex user queries. Now heavily uses PersonalAssistantAgent.
    """
    def __init__(self, memory_store, agent_router, rewriter_agent, post_processor, thread_contexts_ref, persistent_db, orchestrator_agent, available_tools, orchestrated_agents):
        self.memory = memory_store
        self.router = agent_router
        self.rewriter = rewriter_agent
        self.post_processor = post_processor
        self.thread_contexts = thread_contexts_ref
        self.persistent_db = persistent_db
        self.orchestrator_agent = orchestrator_agent # The new PersonalAssistantAgent
        self.available_tools = available_tools # Pass the TOOL_REGISTRY
        self.orchestrated_agents = orchestrated_agents # Pass the list of agents the orchestrator can use

    def _get_thread_summary_internal(self, user_id):
        return get_thread_summary(user_id)

    def _update_thread_internal(self, user_id, message):
        update_thread(user_id, message)

    def _get_persistent_memory_context(self, user_id, prompt_keywords=None, limit=5):
        UserQuery = Query()
        # Retrieve all for user, then filter/sort
        results = self.persistent_db.search(UserQuery.user_id == user_id)
        if results:
            results.sort(key=lambda x: datetime.datetime.fromisoformat(x.get('timestamp', '1970-01-01T00:00:00Z').replace('Z', '+00:00')), reverse=True)
            relevant_context = []
            for item in results[:limit]:
                # Prefer 'prompt' and 'response' directly if available from recent saves
                user_prompt = item.get('prompt')
                ai_response = item.get('response')

                if user_prompt and ai_response:
                    interaction_summary = f"- Past Interaction ({item.get('timestamp')}):\n  User: '{user_prompt}'\n  AI: '{ai_response}'"
                else:
                    # Fallback for older entries or different formats
                    interaction_summary = f"- Past Interaction ({item.get('timestamp')}): {item.get('value', {}).get('prompt', 'N/A')} -> {item.get('value', {}).get('response', 'N/A')}"
                
                relevant_context.append(interaction_summary)
            return "\n".join(relevant_context)
        return "No relevant past interactions found in persistent memory."


    def execute_chain(self, user_id, raw_prompt, persist_memory=False, speak_output=False, voice="Rachel"):
        log_event("Chain Execution Initiated", {"user_id": user_id, "prompt": raw_prompt})
        self._update_thread_internal(user_id, f"User: {raw_prompt}")

        context_summary = self._get_thread_summary_internal(user_id)
        keywords_for_memory = post_processor.tag_keywords(raw_prompt)
        persistent_memory_context = self._get_persistent_memory_context(user_id, prompt_keywords=keywords_for_memory)
        combined_context = f"Short-term chat history:\n{context_summary}\n\nLong-term user memory:\n{persistent_memory_context}"

        # The orchestrator is the first point of contact for complex prompts
        # It handles the prompt refinement internally as part of its planning
        orchestrator_response, execution_details = self.orchestrator_agent.plan_and_execute(
            user_id, raw_prompt, combined_context, self.orchestrated_agents, self.available_tools
        )
        
        agent_name = self.orchestrator_agent.__class__.__name__ # Always PersonalAssistantAgent now for orchestration
        final_response = orchestrator_response

        # Post-process the final response from the orchestrator
        tags = post_processor.tag_keywords(final_response)
        sentiment = post_processor.sentiment_score(final_response)
        audio_base64 = None

        if speak_output:
            try:
                audio_content = text_to_speech(final_response, voice=voice)
                audio_base64 = base64.b64encode(audio_content).decode("utf-8")
            except Exception as e:
                log_event("Audio Generation Failed", {"error": str(e), "text_len": len(final_response)})
                audio_base64 = None


        self._update_thread_internal(user_id, f"Agent ({agent_name}): {final_response}")
        log_event("Orchestrated Response Generated", {"agent": agent_name, "response_len": len(final_response)})

        if persist_memory:
            db_persistent.insert({
                "user_id": user_id,
                "prompt": raw_prompt,
                "refined_prompt": "Handled by orchestrator", # Orchestrator does internal refinement
                "response": final_response,
                "agent": agent_name,
                "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                "sentiment": sentiment,
                "keywords": tags,
                "context_summary_used": context_summary,
                "persistent_memory_used": persistent_memory_context,
                "execution_details": execution_details # Store the multi-step execution log
            })
            log_event("Interaction Saved to Persistent TinyDB", {"user_id": user_id})

        return {
            "user_id": user_id,
            "raw_prompt": raw_prompt,
            "refined_prompt": "Orchestrator handled prompt refinement internally.",
            "context_summary": context_summary,
            "persistent_memory_context_used": persistent_memory_context,
            "agent_used": agent_name,
            "final_response": final_response,
            "sentiment": sentiment,
            "keywords": tags,
            "audio_mpeg_base64": audio_base64,
            "execution_details": execution_details # Return details of the orchestration for debugging/transparency
        }


class Router:
    """Routes prompts to the PersonalAssistantAgent for orchestration."""
    def route(self, prompt):
        # With the PersonalAssistantAgent, most complex queries will route there.
        # Simple, direct queries might still bypass for efficiency if logic is defined,
        # but for max "power", route all through the orchestrator.
        # For this version, let's simplify: almost everything goes to the PersonalAssistantAgent.
        lowered = prompt.lower()
        
        # If it's explicitly a rewriter request, send it to rewriter
        if any(word in lowered for word in ["rewrite", "rephrase", "clarify", "optimize prompt"]):
            return rewriter_agent
        
        # Otherwise, for anything else, send to the orchestrator
        return personal_assistant_agent


router = Router()

# Initialize the SentientChainExecutor with the new orchestrator and tool/agent lists
sentient_executor = SentientChainExecutor(
    memory_store=datastore,
    agent_router=router,
    rewriter_agent=rewriter_agent,
    post_processor=post_processor,
    thread_contexts_ref=thread_contexts,
    persistent_db=db_persistent,
    orchestrator_agent=personal_assistant_agent, # The new powerful AI
    available_tools=TOOL_REGISTRY,               # The tools it can use
    orchestrated_agents=ORCHESTRATED_AGENTS      # The agents it can orchestrate
)

# ========== MINI ROUTES FOR HACK FUN ==========
@app.route("/")
def root():
    return jsonify({"status": "Neuronexus AI Engine operational", "agents": ["Finance", "Mind", "Creative", "VideoCoach", "Rewriter", "PersonalAssistant"], "tools": list(TOOL_REGISTRY.keys())})

@app.route("/finance", methods=["POST"])
@require_json
def route_finance():
    # This route is now mostly for direct testing of the FinanceAgent,
    # complex requests will go via /ai/chain
    data = request.get_json()
    query = data.get("prompt")
    if not query:
        return error_response("Missing prompt.")
    response = finance_agent.respond(query)
    return jsonify({"reply": response})

@app.route("/mind", methods=["POST"])
@require_json
def route_mind():
    # Direct route for MindAgent
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
    voice = data.get("voice", "Rachel")
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
        "version": "0.1c" # Updated version for major changes
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
        # Note: 'value' here is a direct save, not an orchestrated interaction.
        # For orchestrated interactions, data is saved in SentientChainExecutor.
        db_persistent.insert({"user_id": user_id, "key": key, "value": value, "timestamp": datetime.datetime.utcnow().isoformat() + "Z"})
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
            return jsonify({"value": result[-1].get("value")})
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
    Primary endpoint for the enhanced Sentient Chain Executor to handle complex multi-agent interactions.
    All complex user prompts should be sent here.
    """
    data = request.get_json()
    user_id = data.get("user_id")
    raw_prompt = data.get("prompt")
    persist_memory = data.get("persist_memory", False)
    speak_output = data.get("speak_output", False)
    voice = data.get("voice", "Rachel")

    if not all([user_id, raw_prompt]):
        return error_response("Missing user_id or prompt.")

    try:
        # Route through the orchestrator for complex decision-making
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

@app.route("/user/register", methods=["POST"])
@require_json
def register_user():
    """
    Registers a new user (in-memory for this stub).
    """
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")
    wallet_address = data.get("wallet_address")

    if not all([username, password]):
        return error_response("Missing username or password.")

    if username in users:
        return error_response("Username already exists.", 409)

    user_id = hashlib.sha256(username.encode()).hexdigest()
    auth_token = generate_token()

    users[username] = {
        "user_id": user_id,
        "password_hash": hashlib.sha256(password.encode()).hexdigest(),
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
    }), 201

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
        return error_response("Invalid username or password.", 401)

    if user_info["password_hash"] != hashlib.sha256(password.encode()).hexdigest():
        return error_response("Invalid username or password.", 401)

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
    if not OPENAI_API_KEY:
        logging.error("OPENAI_API_KEY environment variable not set.")
    if not ELEVENLABS_API_KEY:
        logging.warning("ELEVENLABS_API_KEY environment variable not set. Voice features will be disabled.")

    app.run(debug=True, host='0.0.0.0', port=5000)
