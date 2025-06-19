# Standard Library Imports
import os
import datetime
import base64
import logging
import hashlib
import random
import json
import uuid # For more robust token generation
import time # For system_pulse uptime

# Third-Party Library Imports
from flask import Flask, request, jsonify
from functools import wraps
from tinydb import TinyDB, Query
from openai import OpenAI # For OpenAI API calls
from pydantic import BaseModel, Field # For request data validation
import requests # For making HTTP requests (e.g., to Algorand, if not using SDK fully)
import bcrypt # For password hashing
# import algosdk # If using the official Algorand SDK more extensively
# from elevenlabs import generate, set_api_key # For ElevenLabs, ensure this is the correct import for your version
from elevenlabs.client import ElevenLabs # Newer ElevenLabs client

# Tokenizer for OpenAI
import tiktoken

# --- Configuration ---
# Load environment variables
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ALGORAND_NODE_URL = os.getenv("ALGORAND_NODE_URL", "https://testnet-api.algonode.cloud")
ALGORAND_EXPLORER_URL = os.getenv("ALGORAND_EXPLORER_URL", "https://testnet.explorer.algorand.org") # Example

# --- Global Initialization ---
app = Flask(__name__)
start_time = datetime.datetime.utcnow()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize OpenAI client
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize ElevenLabs client (assuming newer client library structure)
# If using an older version, adjust imports/usage
elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)


# In-memory storage for chat threads (NOT for production scaling)
thread_contexts = {}

# Initialize TinyDB for persistent memory
db_persistent = TinyDB('persistent_memory.json')

# In-memory user store (NOT for production use, replace with a real database)
users = {}

# --- Utility Functions ---

def log_event(event_name, details=None):
    """Logs structured events."""
    if details is None:
        details = {}
    logging.info(f"EVENT: {event_name} | DETAILS: {json.dumps(details)}")

def error_response(message, status_code=400):
    """Generates a consistent error response."""
    log_event("API_Error", {"message": message, "status_code": status_code})
    return jsonify({"error": message}), status_code

def require_json(f):
    """Decorator to ensure request is JSON."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not request.is_json:
            return error_response("Request must be JSON", 400)
        return f(*args, **kwargs)
    return decorated_function

def generate_token():
    """Generates a secure random token."""
    return uuid.uuid4().hex

# --- AI Completion Function ---
def ai_completion(
    messages,
    model="gpt-4o", # Updated to a more powerful model as per original code
    temperature=0.7,
    max_tokens=4000,
    json_mode=False,
    tools=None, # For tool calling
    tool_choice="auto", # For tool calling
):
    """
    Handles AI completion requests with OpenAI's API, supporting JSON mode and tool calling.
    """
    if not OPENAI_API_KEY:
        log_event("AI_Completion_Error", {"message": "OPENAI_API_KEY not set."})
        raise RuntimeError("OpenAI API key not configured.")

    # The 'headers' part is implicitly handled by the OpenAI client.
    # No need to manually construct headers for openai_client.chat.completions.create
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = tool_choice

    try:
        log_event("OpenAI_Request", {"model": model, "json_mode": json_mode, "message_count": len(messages)})
        response = openai_client.chat.completions.create(**payload)

        # Log full response for debugging
        # log_event("OpenAI_Full_Response", {"response": response.model_dump_json()})

        # Extract content based on response structure
        if response.choices and response.choices[0].message:
            message = response.choices[0].message
            content = message.content

            if message.tool_calls:
                log_event("OpenAI_Tool_Calls", {"tool_calls": [tc.function.name for tc in message.tool_calls]})
                return content, message.tool_calls
            else:
                return content, None # No tool calls

        log_event("OpenAI_Completion_Empty_Response", {"response_object": response.model_dump_json()})
        return "No response generated.", None

    except Exception as e:
        log_event("OpenAI_Completion_Failure", {"error": str(e), "payload": payload})
        raise RuntimeError(f"OpenAI API error: {e}")

# --- Text to Speech Function ---
def text_to_speech(text, voice="Rachel"):
    """Converts text to speech using ElevenLabs."""
    if not ELEVENLABS_API_KEY:
        log_event("TTS_Error", {"message": "ELEVENLABS_API_KEY not set."})
        raise RuntimeError("ElevenLabs API key not configured. Voice features disabled.")

    try:
        # Using the newer elevenlabs.client.ElevenLabs client
        # voice_id_map is defined globally below, after the function
        audio = elevenlabs_client.generate(
            text=text,
            voice=voice_id_map.get(voice, '21m00Tzpb8CmLzInGRDq'), # Default to Rachel
            model="eleven_multilingual_v2" # Example model
        )
        # The 'generate' method often returns an iterator or bytes directly
        # If it's an iterator, you might need to combine chunks
        # For simplicity, assuming it directly returns bytes or a BytesIO object
        # If 'audio' is an iterator, use: audio_bytes = b"".join(audio)
        return audio # Return bytes

    except Exception as e:
        log_event("ElevenLabs_Generation_Error", {"error": str(e), "text_len": len(text)})
        raise RuntimeError(f"Failed to generate speech: {e}")

voice_id_map = {
    "Rachel": "21m00Tzpb8CmLzInGRDq",
    "Adam": "pNlFjC7aYgLqgOa055p0", # Example additional voice
    # Add more voice mappings if needed
}

# --- Algorand Integration Stubs ---
def verify_algorand_wallet(address):
    """
    Verifies if an Algorand address is valid.
    # TODO: Implement actual Algorand SDK validation or API call
    """
    if not address or len(address) != 58: # Basic length check for mainnet/testnet addresses
        return False
    # This is a very basic check; a real check would use algosdk.encoding.is_valid_address
    try:
        # from algosdk.encoding import is_valid_address
        # return is_valid_address(address)
        return True # Placeholder for now
    except Exception:
        return False

def get_account_balance(wallet_address):
    """
    Retrieves the Algorand account balance.
    # TODO: Implement actual Algorand SDK or API call
    """
    if not ALGORAND_NODE_URL:
        log_event("Algorand_Error", {"message": "ALGORAND_NODE_URL not set."})
        return "Algorand node URL not configured."

    # Example using requests, but you should use the official algosdk for robust interaction
    try:
        # from algosdk.v2client import algod
        # algod_client = algod.AlgodClient("your_algod_token", ALGORAND_NODE_URL)
        # account_info = algod_client.account_info(wallet_address)
        # return account_info.get("amount") / 1_000_000 # Convert microAlgos to Algos
        response = requests.get(f"{ALGORAND_NODE_URL}/v2/accounts/{wallet_address}")
        response.raise_for_status()
        data = response.json()
        # Algorand amounts are in microAlgos
        balance_micro_algos = data.get('amount', 0)
        balance_algos = balance_micro_algos / 1_000_000
        log_event("Algorand_Balance_Fetched", {"wallet_address": wallet_address, "balance": balance_algos})
        return f"{balance_algos} ALGO"
    except requests.exceptions.RequestException as e:
        log_event("Algorand_API_Error", {"error": str(e), "wallet_address": wallet_address})
        return f"Error fetching balance: {e}"
    except Exception as e:
        log_event("Algorand_Balance_Error", {"error": str(e), "wallet_address": wallet_address})
        return f"An unexpected error occurred: {e}"


# --- Memory Management ---
class DataStore:
    """In-memory key-value store for simple short-term memory."""
    def __init__(self):
        self.store = {}

    def remember(self, user_id, key, value):
        if user_id not in self.store:
            self.store[user_id] = {}
        self.store[user_id][key] = value
        log_event("DataStore_Remembered", {"user_id": user_id, "key": key})

    def recall(self, user_id, key):
        value = self.store.get(user_id, {}).get(key)
        log_event("DataStore_Recalled", {"user_id": user_id, "key": key, "found": bool(value)})
        return value

datastore = DataStore() # Initialize DataStore

def get_thread_summary(user_id):
    """Summarizes a user's chat thread for context."""
    thread = thread_contexts.get(user_id, [])
    if not thread:
        return "No recent chat history."

    # Use AI to summarize the thread
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Summarize the following chat conversation concisely, focusing on key topics and decisions. If the conversation is empty, say 'No recent chat history.'"},
        {"role": "user", "content": "\n".join(thread)}
    ]
    try:
        summary, _ = ai_completion(messages, model="gpt-3.5-turbo", max_tokens=500) # Use a cheaper model for summary
        log_event("Thread_Summary_Generated", {"user_id": user_id, "summary_len": len(summary)})
        return summary
    except Exception as e:
        log_event("Thread_Summary_Error", {"user_id": user_id, "error": str(e)})
        return "Failed to summarize chat history due to an AI error."

def update_thread(user_id, message):
    """Updates the in-memory chat thread for a user."""
    if user_id not in thread_contexts:
        thread_contexts[user_id] = []
    thread_contexts[user_id].append(message)
    # Keep thread relatively short to avoid excessive token usage for summaries
    if len(thread_contexts[user_id]) > 20: # Keep last 20 messages
        thread_contexts[user_id] = thread_contexts[user_id][-20:]
    log_event("Thread_Updated", {"user_id": user_id, "message_len": len(message)})

# --- Tool Definitions ---
class Tool:
    """Base class for all tools."""
    def __init__(self, name, description, parameters_schema):
        self.name = name
        self.description = description
        self.parameters_schema = parameters_schema

    def execute(self, **kwargs):
        raise NotImplementedError("Each tool must implement its execute method.")

class KnowledgeBaseTool(Tool):
    def __init__(self):
        super().__init__(
            name="KnowledgeBase",
            description="Provides access to simulated financial data and market news. Use for stock prices, company info, or general market updates.",
            parameters_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The specific query for financial data or news."},
                    "data_type": {"type": "string", "enum": ["stock_data", "market_news"], "description": "The type of data to retrieve (stock_data or market_news)."}
                },
                "required": ["query", "data_type"]
            }
        )

    def execute(self, query, data_type):
        if data_type == "stock_data":
            return self._get_stock_data(query)
        elif data_type == "market_news":
            return self._get_market_news(query)
        return "Invalid data_type for KnowledgeBaseTool."

    def _get_stock_data(self, symbol):
        # TODO: Replace with actual stock market API integration (e.g., Alpha Vantage, Yahoo Finance API)
        log_event("Tool_Execution", {"tool": "KnowledgeBase", "action": "get_stock_data", "symbol": symbol})
        mock_data = {
            "AAPL": {"price": 175.50, "volume": "70M", "change": "+1.2%", "currency": "USD"},
            "GOOG": {"price": 1700.00, "volume": "25M", "change": "-0.5%", "currency": "USD"},
            "MSFT": {"price": 420.00, "volume": "50M", "change": "+0.8%", "currency": "USD"},
        }
        data = mock_data.get(symbol.upper())
        if data:
            return f"Stock data for {symbol.upper()}: Price {data['price']} {data['currency']}, Volume {data['volume']}, Change {data['change']}."
        return f"Could not find stock data for {symbol.upper()}."

    def _get_market_news(self, topic):
        # TODO: Replace with actual news API integration (e.g., NewsAPI, GNews API)
        log_event("Tool_Execution", {"tool": "KnowledgeBase", "action": "get_market_news", "topic": topic})
        mock_news = {
            "tech": "Tech giants announce new AI advancements, boosting sector confidence.",
            "finance": "Global markets react to interest rate hike speculation; inflation concerns persist.",
            "energy": "Oil prices stabilize after recent volatility, focusing on supply chain resilience."
        }
        news = mock_news.get(topic.lower())
        if news:
            return f"Latest news on {topic}: {news}"
        return f"No specific news found for {topic}."

class CalculatorTool(Tool):
    def __init__(self):
        super().__init__(
            name="Calculator",
            description="Performs basic arithmetic calculations.",
            parameters_schema={
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "The mathematical expression to evaluate."}
                },
                "required": ["expression"]
            }
        )

    def execute(self, expression):
        log_event("Tool_Execution", {"tool": "Calculator", "expression": expression})
        try:
            # WARNING: Using eval() directly can be a security risk if input is not controlled.
            # For a production system, consider a safer math expression parser/evaluator.
            result = eval(expression)
            return f"The result of '{expression}' is {result}."
        except Exception as e:
            return f"Error calculating '{expression}': {e}"

class CalendarTool(Tool):
    def __init__(self):
        super().__init__(
            name="Calendar",
            description="Manages calendar events (create, view).",
            parameters_schema={
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["create_event", "get_events"], "description": "The calendar action to perform."},
                    "event_details": {"type": "string", "description": "Details for creating an event (e.g., 'Meeting with John tomorrow at 10 AM'). Required for create_event."},
                    "date": {"type": "string", "description": "Date to get events for (e.g., 'today', 'next week', '2025-06-25'). Required for get_events."}
                },
                "required": ["action"]
            }
        )

    def execute(self, action, **kwargs):
        log_event("Tool_Execution", {"tool": "Calendar", "action": action, "kwargs": kwargs})
        if action == "create_event":
            return self._create_event(kwargs.get("event_details"))
        elif action == "get_events":
            return self._get_events(kwargs.get("date"))
        return "Invalid action for CalendarTool."

    def _create_event(self, event_details):
        # TODO: Integrate with a real calendar API (e.g., Google Calendar API)
        if not event_details:
            return "Event details are required to create an event."
        return f"Event '{event_details}' successfully created on your calendar (simulated)."

    def _get_events(self, date):
        # TODO: Integrate with a real calendar API (e.g., Google Calendar API)
        if not date:
            return "Date is required to get events."
        mock_events = {
            "today": "10:00 AM - Team Sync; 2:00 PM - Project Review.",
            "tomorrow": "9:00 AM - Client Call; 1:00 PM - Deep Work Session.",
            "2025-06-25": "No events scheduled."
        }
        events = mock_events.get(date.lower(), "No events found for that date.")
        return f"Events for {date}: {events}"

class WebSearchTool(Tool):
    def __init__(self):
        super().__init__(
            name="WebSearch",
            description="Performs a web search to find information online. Use for general knowledge queries.",
            parameters_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query."}
                },
                "required": ["query"]
            }
        )

    def execute(self, query):
        # TODO: Integrate with a real web search API (e.g., Google Custom Search API, SerpAPI, Bing Search API)
        log_event("Tool_Execution", {"tool": "WebSearch", "query": query})
        mock_results = {
            "latest AI news": "Researchers at Google DeepMind announced a breakthrough in multimodal AI models.",
            "weather in London": "The weather in London tomorrow will be partly cloudy with a high of 20°C.",
            "history of AI": "Artificial intelligence has its roots in the 1950s with pioneers like Alan Turing."
        }
        # Simulate relevance
        for key, value in mock_results.items():
            if query.lower() in key.lower() or key.lower() in query.lower():
                return f"Search result for '{query}': {value}"
        return f"No direct search result found for '{query}' (simulated)."

# Initialize tools
TOOL_REGISTRY = {
    "knowledge_base": KnowledgeBaseTool(),
    "calculator": CalculatorTool(),
    "calendar": CalendarTool(),
    "web_search": WebSearchTool()
}

# --- Agent Definitions ---
class BaseAgent:
    """Abstract base class for all AI agents."""
    def __init__(self, name, description, system_prompt, model="gpt-4o"):
        self.name = name
        self.description = description
        self.system_prompt = system_prompt
        self.model = model

    def respond(self, user_prompt, chat_history=None, **kwargs):
        """Generates a response using AI completion."""
        messages = [{"role": "system", "content": self.system_prompt}]
        if chat_history:
            # Assuming chat_history is a list of {"role": ..., "content": ...}
            messages.extend(chat_history)
        messages.append({"role": "user", "content": user_prompt})

        try:
            response_content, _ = ai_completion(messages, model=self.model, **kwargs)
            log_event("Agent_Response", {"agent": self.name, "prompt_len": len(user_prompt), "response_len": len(response_content)})
            return response_content
        except RuntimeError as e:
            log_event("Agent_Response_Error", {"agent": self.name, "error": str(e)})
            return f"I'm sorry, I encountered an error while processing your request: {e}"

class FinanceAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="FinanceAgent",
            description="Specializes in financial queries, stock data, market analysis, and investment advice.",
            system_prompt="You are a financial expert. Provide accurate and concise information regarding stocks, markets, and investments. Do not give direct financial advice, but offer factual information and general insights. Use the KnowledgeBase tool for specific financial data."
        )

    # In this orchestrated system, the orchestrator will call tools based on its plan.
    # The agent's respond method here primarily provides the AI's persona and general knowledge.
    # Tool execution happens external to this specific agent's 'respond'
    def respond(self, user_prompt, chat_history=None):
        return super().respond(user_prompt, chat_history)

class MindAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="MindAgent",
            description="Focuses on mental well-being, mindfulness, emotional support, and cognitive exercises.",
            system_prompt="You are a compassionate and insightful guide for mental well-being. Offer mindfulness techniques, emotional support, and cognitive behavioral insights. Avoid diagnosing or replacing professional medical advice."
        )

    def respond(self, user_prompt, chat_history=None):
        return super().respond(user_prompt, chat_history)

class CreativeAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="CreativeAgent",
            description="Generates creative content like stories, poems, blog ideas, and social media posts.",
            system_prompt="You are a highly creative content generator. Your goal is to produce engaging and original text, whether it's for social media, stories, or marketing. Be imaginative and adapt to the requested style."
        )

    def reddit_post(self, topic):
        # TODO: Implement more sophisticated post generation
        prompt = f"Generate a creative and engaging Reddit post about '{topic}' for a general audience. Include a catchy title and some sub-points."
        return super().respond(prompt)

    def meme_caption(self, topic):
        # TODO: Implement more sophisticated caption generation
        prompt = f"Generate a humorous and concise meme caption about '{topic}'."
        return super().respond(prompt)

    def blog_idea(self, niche):
        # TODO: Implement more sophisticated blog idea generation
        prompt = f"Generate a unique and compelling blog post idea for the niche: '{niche}'. Include a catchy title and 3-5 bullet points for content."
        return super().respond(prompt)

class VideoCoach(BaseAgent):
    def __init__(self):
        super().__init__(
            name="VideoCoach",
            description="Simulates a video coaching session, providing guidance and feedback.",
            system_prompt="You are an encouraging and insightful video coach. You provide constructive feedback, ask probing questions, and guide the user through simulated scenarios as if in a live video call. Focus on active listening and supportive guidance."
        )

    def simulate_session(self, prompt_text):
        # TODO: This is a direct response; for a true simulation, it would involve
        # more back-and-forth or integration with video analysis.
        prompt = f"Simulate a coaching response to the following user input, as if in a video call: '{prompt_text}'. Provide empathetic understanding, ask a clarifying question, and offer a small piece of actionable advice."
        return super().respond(prompt)

class RewriterAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="RewriterAgent",
            description="Refines and rewrites prompts or text for clarity, conciseness, or a specific tone.",
            system_prompt="You are an expert text rewriter. Your purpose is to refine, rephrase, or optimize given text for clarity, conciseness, or to match a specified tone (e.g., professional, casual, persuasive). Always improve the input significantly."
        )

    def respond(self, text_to_rewrite, tone=None):
        rewrite_prompt = f"Please rewrite the following text. Make it more clear and concise."
        if tone:
            rewrite_prompt += f" Adopt a {tone} tone."
        rewrite_prompt += f"\n\nText: '{text_to_rewrite}'"
        return super().respond(rewrite_prompt)

class PostProcessor:
    """Processes AI output for additional insights."""
    def tag_keywords(self, text):
        prompt = f"Extract up to 5 main keywords or tags from the following text, separated by commas. Respond only with the keywords.\n\nText: '{text}'"
        try:
            keywords_str, _ = ai_completion([{"role": "user", "content": prompt}], model="gpt-3.5-turbo", max_tokens=100)
            return [k.strip() for k in keywords_str.split(',') if k.strip()]
        except Exception as e:
            log_event("PostProcessor_Keyword_Error", {"error": str(e), "text_len": len(text)})
            return []

    def sentiment_score(self, text):
        prompt = f"Analyze the sentiment of the following text and classify it as positive, negative, or neutral. Respond only with one word: 'positive', 'negative', or 'neutral'.\n\nText: '{text}'"
        try:
            sentiment, _ = ai_completion([{"role": "user", "content": prompt}], model="gpt-3.5-turbo", max_tokens=20)
            sentiment = sentiment.strip().lower()
            if sentiment in ["positive", "negative", "neutral"]:
                return sentiment
            return "unknown"
        except Exception as e:
            log_event("PostProcessor_Sentiment_Error", {"error": str(e), "text_len": len(text)})
            return "unknown"

# Initialize agents and post-processor
finance_agent = FinanceAgent()
mind_agent = MindAgent()
creative_agent = CreativeAgent()
video_coach = VideoCoach()
rewriter_agent = RewriterAgent()
post_processor = PostProcessor()

# Define agents that the orchestrator can call
ORCHESTRATED_AGENTS = {
    "FinanceAgent": finance_agent,
    "MindAgent": mind_agent,
    "CreativeAgent": creative_agent,
    "VideoCoach": video_coach,
    "RewriterAgent": rewriter_agent,
}

class PersonalAssistantAgent(BaseAgent):
    """
    The orchestrator agent responsible for planning and executing multi-step tasks
    using other agents and tools.
    """
    def __init__(self):
        super().__init__(
            name="PersonalAssistantAgent",
            description="Your primary AI orchestrator. Capable of understanding complex requests, breaking them into steps, and delegating to specialized agents and tools. It can also synthesize information and provide a final comprehensive answer.",
            system_prompt="""You are an advanced AI orchestrator named Neuronexus, designed to assist users by intelligently planning and executing complex tasks.
Your capabilities include:
1.  **Understanding user intent**: Fully grasp what the user wants, even if it's multi-faceted.
2.  **Accessing available tools**: Use tools to gather information or perform actions.
3.  **Delegating to specialized agents**: Route specific parts of a request to the most appropriate specialized agent.
4.  **Creating a multi-step plan**: Formulate a JSON plan that outlines the sequence of actions.
5.  **Executing the plan**: Follow through with each step, integrating results.
6.  **Synthesizing a final, comprehensive response**: Provide a clear, concise, and helpful answer to the user.

**Available Tools:**
Each tool has a 'name', 'description', and 'parameters_schema'.
{tools_json}

**Available Specialized Agents:**
Each agent has a 'name' and 'description'.
{agents_json}

**Planning Instructions:**
Your response MUST be a JSON object containing a 'plan' array. Each item in the 'plan' array must be a JSON object with the following structure:
-   `step_name`: A brief, descriptive name for the step (e.g., "Check Stock Price", "Summarize News").
-   `step_type`: "tool_use" or "agent_call" or "final_response".
-   `tool_name`: (Required if `step_type` is "tool_use") The name of the tool to use from the TOOL_REGISTRY.
-   `tool_parameters`: (Required if `step_type` is "tool_use") A dictionary of parameters for the tool's 'execute' method, matching its 'parameters_schema'.
-   `agent_name`: (Required if `step_type` is "agent_call") The name of the agent to call from ORCHESTRATED_AGENTS.
-   `agent_prompt`: (Required if `step_type` is "agent_call") The prompt to send to the selected agent.
-   `response_synthesis_prompt`: (Required if `step_type` is "final_response") The prompt to use for generating the final response, incorporating results from previous steps using placeholders like `{{step_name_output}}`.

**Example Plan for "What's the stock price of AAPL and tell me something positive to start my day?":**
```json
{{
  "plan": [
    {{
      "step_name": "GetAAPLStock",
      "step_type": "tool_use",
      "tool_name": "KnowledgeBase",
      "tool_parameters": {{
        "query": "AAPL",
        "data_type": "stock_data"
      }}
    }},
    {{
      "step_name": "GetPositiveThought",
      "step_type": "agent_call",
      "agent_name": "MindAgent",
      "agent_prompt": "Give me a short, positive thought to start the day."
    }},
    {{
      "step_name": "FinalResponse",
      "step_type": "final_response",
      "response_synthesis_prompt": "Based on the following:\nAAPL Stock: {{GetAAPLStock_output}}\nPositive Thought: {{GetPositiveThought_output}}\n\nProvide a comprehensive and uplifting response."
    }}
  ]
}}
   )

def plan_and_execute(self, user_id, raw_prompt, combined_context, orchestrated_agents, available_tools):
    log_event("Orchestrator_Plan_Initiated", {"user_id": user_id, "prompt": raw_prompt})

    # Format available tools and agents for the prompt
    tools_list = [{"name": t.name, "description": t.description, "parameters_schema": t.parameters_schema} for t in available_tools.values()]
    agents_list = [{"name": a.name, "description": a.description} for a in orchestrated_agents.values()]

    tools_json = json.dumps(tools_list, indent=2)
    agents_json = json.dumps(agents_list, indent=2)

    planning_system_prompt = self.system_prompt.format(tools_json=tools_json, agents_json=agents_json)

    messages = [
        {"role": "system", "content": planning_system_prompt},
        {"role": "user", "content": f"User's request:\n'{raw_prompt}'\n\nAdditional Context:\n{combined_context}"}
    ]

    plan_json_str = ""
    execution_log = []
    try:
        # Get the plan from the AI
        plan_json_str, _ = ai_completion(messages, model=self.model, json_mode=True)
        plan_data = json.loads(plan_json_str)
        plan = plan_data.get("plan", [])
        log_event("Orchestrator_Plan_Received", {"user_id": user_id, "plan": plan})
        execution_log.append({"type": "plan", "plan": plan})

    except json.JSONDecodeError as e:
        error_msg = f"Orchestrator failed to generate valid JSON plan: {e}. Raw response: {plan_json_str[:500]}"
        log_event("Orchestrator_JSON_Error", {"user_id": user_id, "error": error_msg})
        return f"I'm sorry, I couldn't understand the plan generated. Please try rephrasing your request. (Error: Invalid JSON from AI)", execution_log
    except RuntimeError as e:
        error_msg = f"Orchestrator AI planning failed: {e}"
        log_event("Orchestrator_Planning_Error", {"user_id": user_id, "error": error_msg})
        return f"I'm sorry, I couldn't create a plan to fulfill your request. (Error: {e})", execution_log
    except Exception as e:
        error_msg = f"An unexpected error occurred during planning: {e}"
        log_event("Orchestrator_Unexpected_Planning_Error", {"user_id": user_id, "error": error_msg})
        return f"An unexpected error occurred while planning: {e}", execution_log

    step_outputs = {}
    final_response_prompt = ""

    for i, step in enumerate(plan):
        step_name = step.get("step_name", f"step_{i}")
        step_type = step.get("step_type")
        log_event("Orchestrator_Executing_Step", {"user_id": user_id, "step_name": step_name, "step_type": step_type})
        execution_log.append({"type": "execution_step", "step_name": step_name, "step_type": step_type})

        try:
            if step_type == "tool_use":
                tool_name = step.get("tool_name")
                tool_parameters = step.get("tool_parameters", {})
                if tool_name in available_tools:
                    tool_output = available_tools[tool_name].execute(**tool_parameters)
                    step_outputs[f"{step_name}_output"] = tool_output
                    execution_log.append({"type": "tool_output", "step_name": step_name, "output": tool_output})
                else:
                    tool_output = f"Error: Tool '{tool_name}' not found."
                    step_outputs[f"{step_name}_output"] = tool_output
                    execution_log.append({"type": "tool_error", "step_name": step_name, "error": tool_output})

            elif step_type == "agent_call":
                agent_name = step.get("agent_name")
                agent_prompt = step.get("agent_prompt")
                if agent_name in orchestrated_agents:
                    agent_response = orchestrated_agents[agent_name].respond(agent_prompt)
                    step_outputs[f"{step_name}_output"] = agent_response
                    execution_log.append({"type": "agent_output", "step_name": step_name, "output": agent_response})
                else:
                    agent_response = f"Error: Agent '{agent_name}' not found."
                    step_outputs[f"{step_name}_output"] = agent_response
                    execution_log.append({"type": "agent_error", "step_name": step_name, "error": agent_response})

            elif step_type == "final_response":
                final_response_prompt = step.get("response_synthesis_prompt", "Please provide a final answer based on the information gathered.")
                log_event("Orchestrator_Final_Response_Planned", {"user_id": user_id, "prompt": final_response_prompt})
                # Break loop, as final response step is usually last
                break
            else:
                error_message = f"Unknown step_type: {step_type}"
                log_event("Orchestrator_Unknown_Step_Type", {"user_id": user_id, "step_name": step_name, "step_type": step_type})
                step_outputs[f"{step_name}_output"] = error_message
                execution_log.append({"type": "step_error", "step_name": step_name, "error": error_message})

        except Exception as e:
            error_message = f"Error during step '{step_name}': {e}"
            log_event("Orchestrator_Step_Execution_Error", {"user_id": user_id, "step_name": step_name, "error": error_message})
            step_outputs[f"{step_name}_output"] = error_message
            execution_log.append({"type": "step_error", "step_name": step_name, "error": error_message})
            # Decide if you want to stop execution on first error or try to continue
            # For now, let's just log and continue for robustness.

    # Synthesize final response
    if not final_response_prompt:
        # Fallback if no final_response step was specified in the plan
        final_response_prompt = "No specific final response prompt was provided by the plan. Synthesize a response based on the outputs of the previous steps:\n"
        for k, v in step_outputs.items():
            final_response_prompt += f"{k}: {v}\n"

    # Replace placeholders in the final response prompt with actual outputs
    for placeholder, value in step_outputs.items():
        final_response_prompt = final_response_prompt.replace(f"{{{{{placeholder}}}}}", str(value))

    try:
        # The system prompt here for the final response can be generic
        final_messages = [
            {"role": "system", "content": "You are a helpful assistant. Consolidate the provided information and answer the user's request comprehensively and clearly. Address all parts of the original request."},
            {"role": "user", "content": final_response_prompt}
        ]
        final_answer, _ = ai_completion(final_messages, model=self.model)
        log_event("Orchestrator_Final_Response_Generated", {"user_id": user_id, "response_len": len(final_answer)})
        execution_log.append({"type": "final_response", "output": final_answer})
        return final_answer, execution_log
    except Exception as e:
        error_msg = f"Error synthesizing final response: {e}"
        log_event("Orchestrator_Final_Synthesis_Error", {"user_id": user_id, "error": error_msg})
        return f"I managed to get some information, but I couldn't synthesize a complete answer: {error_msg}. Intermediate results: {json.dumps(step_outputs)}", execution_log
personal_assistant_agent = PersonalAssistantAgent() # Instantiate the Orchestrator
# Instantiate the Router here, after agents are defined
class Router:
    """Routes prompts to the PersonalAssistantAgent for orchestration or RewriterAgent for specific tasks."""
    def route(self, prompt):
        lowered = prompt.lower()
        if any(word in lowered for word in ["rewrite", "rephrase", "clarify", "optimize prompt"]):
            return rewriter_agent # Route to rewriter if the intent is clearly rewriting
        return personal_assistant_agent # Default to orchestrator for complex queries

router = Router() # Instantiate the router

# Initialize the SentientChainExecutor with the new orchestrator and tool/agent lists
class SentientChainExecutor:
    def __init__(self, memory_store, agent_router, rewriter_agent, post_processor, thread_contexts_ref, persistent_db, orchestrator_agent, available_tools, orchestrated_agents):
        self.memory_store = memory_store # For simple in-memory K-V pairs
        self.agent_router = agent_router
        self.rewriter_agent = rewriter_agent
        self.post_processor = post_processor
        self.thread_contexts = thread_contexts_ref # Reference to the global dict for chat history
        self.persistent_db = persistent_db # TinyDB instance
        self.orchestrator_agent = orchestrator_agent # The main orchestrator (PersonalAssistantAgent)
        self.available_tools = available_tools # Pass the TOOL_REGISTRY
        self.orchestrated_agents = orchestrated_agents # Pass the list of agents the orchestrator can use

    def execute_chain(self, user_id, prompt):
        log_event("Executor_Execution_Started", {"user_id": user_id, "prompt_len": len(prompt)})

        # 1. Routing
        target_agent = self.agent_router.route(prompt)
        log_event("Executor_Routing_Decision", {"user_id": user_id, "target_agent": target_agent.name})

        # Get relevant context (chat history, user data from DataStore, persistent DB)
        chat_history_summary = get_thread_summary(user_id) # Summarize chat history
        user_data = self.memory_store.recall(user_id, "user_profile") # Example of recalling from DataStore
        persistent_user_data = self.persistent_db.search(Query().user_id == user_id) # Example from TinyDB

        combined_context = f"Chat History Summary: {chat_history_summary}\nUser Profile: {user_data}\nPersistent Data: {persistent_user_data}"
        log_event("Executor_Context_Gathered", {"user_id": user_id, "context_len": len(combined_context)})

        raw_response = ""
        execution_log = [] # To capture detailed steps for debugging/auditing

        if target_agent == self.orchestrator_agent:
            # Orchestrator handles complex, multi-step queries
            final_answer, log = self.orchestrator_agent.plan_and_execute(
                user_id=user_id,
                raw_prompt=prompt,
                combined_context=combined_context,
                orchestrated_agents=self.orchestrated_agents,
                available_tools=self.available_tools
            )
            raw_response = final_answer
            execution_log.extend(log)

        elif target_agent == self.rewriter_agent:
            # Rewriter handles specific rewriting tasks directly
            # For simplicity, we'll assume the prompt itself is the text to rewrite for now
            # In a more complex scenario, you might have the LLM extract the text and tone
            raw_response = self.rewriter_agent.respond(prompt)
            execution_log.append({"type": "direct_agent_call", "agent": "RewriterAgent", "response": raw_response})

        else:
            # Fallback for other direct agents if needed, though router currently only sends to orchestrator or rewriter
            raw_response = target_agent.respond(prompt, chat_history=[{"role": "user", "content": prompt}]) # Minimal history
            execution_log.append({"type": "direct_agent_call", "agent": target_agent.name, "response": raw_response})


        # 3. Post-processing (e.g., sentiment, keywords)
        keywords = self.post_processor.tag_keywords(raw_response)
        sentiment = self.post_processor.sentiment_score(raw_response)
        log_event("Executor_Post_Processed", {"user_id": user_id, "keywords": keywords, "sentiment": sentiment})
        execution_log.append({"type": "post_processing", "keywords": keywords, "sentiment": sentiment})

        # 4. Update chat thread
        update_thread(user_id, f"User: {prompt}")
        update_thread(user_id, f"AI: {raw_response}")

        log_event("Executor_Execution_Completed", {"user_id": user_id})
        return raw_response, {"keywords": keywords, "sentiment": sentiment, "log": execution_log}


sentient_executor = SentientChainExecutor(
    memory_store=datastore,
    agent_router=router,
    rewriter_agent=rewriter_agent,
    post_processor=post_processor,
    thread_contexts_ref=thread_contexts,
    persistent_db=db_persistent,
    orchestrator_agent=personal_assistant_agent,
    available_tools=TOOL_REGISTRY,
    orchestrated_agents=ORCHESTRATED_AGENTS
)

# --- Flask Routes ---

class PromptRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    prompt: str = Field(..., min_length=1)
    # Add other fields as needed, e.g., thread_id for specific conversations

@app.route("/health", methods=["GET"])
def health_check():
    uptime = datetime.datetime.utcnow() - start_time
    return jsonify({
        "status": "healthy",
        "uptime": str(uptime),
        "timestamp": datetime.datetime.utcnow().isoformat()
    })

@app.route("/user/register", methods=["POST"])
@require_json
def register_user():
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")

    if not username or not password:
        return error_response("Username and password are required.", 400)

    if username in users:
        return error_response("Username already exists.", 409)

    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    token = generate_token()
    users[username] = {"password": hashed_password, "token": token, "user_id": str(uuid.uuid4())}
    log_event("User_Registered", {"username": username, "user_id": users[username]["user_id"]})
    return jsonify({
        "message": "User registered successfully.",
        "username": username,
        "user_id": users[username]["user_id"],
        "token": token # In a real app, this token would be stored securely and used for auth
    }), 201

@app.route("/user/login", methods=["POST"])
@require_json
def login_user():
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")

    if not username or not password:
        return error_response("Username and password are required.", 400)

    user = users.get(username)
    if user and bcrypt.checkpw(password.encode('utf-8'), user["password"].encode('utf-8')):
        log_event("User_Logged_In", {"username": username, "user_id": user["user_id"]})
        return jsonify({
            "message": "Login successful.",
            "username": username,
            "user_id": user["user_id"],
            "token": user["token"]
        }), 200
    else:
        return error_response("Invalid username or password.", 401)

@app.route("/ai/chain", methods=["POST"])
@require_json
def ai_chain_endpoint():
    try:
        req_data = PromptRequest(**request.get_json())
        user_id = req_data.user_id
        user_prompt = req_data.prompt

        # Execute the AI chain
        response_content, metadata = sentient_executor.execute_chain(user_id, user_prompt)

        return jsonify({
            "response": response_content,
            "metadata": metadata,
            "timestamp": datetime.datetime.utcnow().isoformat()
        }), 200

    except ValueError as e:
        return error_response(f"Invalid request data: {e}", 422)
    except RuntimeError as e:
        return error_response(f"AI chain execution failed: {e}", 500)
    except Exception as e:
        log_event("Unhandled_Error_AI_Chain", {"error": str(e), "request_data": request.get_json()})
        return error_response(f"An unexpected error occurred: {e}", 500)

@app.route("/ai/speak", methods=["POST"])
@require_json
def ai_speak_endpoint():
    data = request.get_json()
    text = data.get("text")
    voice = data.get("voice", "Rachel") # Default voice

    if not text:
        return error_response("Text is required for speech generation.", 400)

    try:
        audio_bytes = text_to_speech(text, voice)
        # Encode audio bytes to base64 for JSON response
        encoded_audio = base64.b64encode(audio_bytes).decode('utf-8')
        return jsonify({
            "audio_base64": encoded_audio,
            "format": "mp3" # ElevenLabs typically returns mp3
        }), 200
    except RuntimeError as e:
        return error_response(str(e), 500)
    except Exception as e:
        log_event("Unhandled_Error_TTS", {"error": str(e), "text_len": len(text)})
        return error_response(f"An unexpected error occurred during speech generation: {e}", 500)

@app.route("/algo/balance", methods=["POST"])
@require_json
def algo_balance_endpoint():
    data = request.get_json()
    wallet_address = data.get("wallet_address")

    if not wallet_address:
        return error_response("Wallet address is required.", 400)

    if not verify_algorand_wallet(wallet_address):
        return error_response("Invalid Algorand wallet address format.", 400)

    balance_info = get_account_balance(wallet_address)
    if "Error" in balance_info:
        return error_response(balance_info, 500) # Propagate error from get_account_balance
    return jsonify({
        "wallet_address": wallet_address,
        "balance": balance_info
    }), 200


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
