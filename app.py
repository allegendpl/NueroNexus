import os
import datetime
import json
import uuid
import base64
import functools
from collections import defaultdict
import time
import requests # For external API calls like ElevenLabs (mocked)
from typing import Dict, Any, List, Tuple, Callable

# External libraries (install via pip: pip install Flask pydantic python-dotenv tinydb bcrypt)
from flask import Flask, request, jsonify
from pydantic import BaseModel, Field, ValidationError
import bcrypt
from tinydb import TinyDB, Query, where

# --- Configuration & Environment Variables (Mocking for demonstration) ---
# In a real application, these would be loaded from .env or similar.
# os.environ["ELEVENLABS_API_KEY"] = "YOUR_ELEVENLABS_API_KEY" # Replace with actual key
# os.environ["GOOGLE_API_KEY"] = "YOUR_GOOGLE_API_KEY" # Replace with actual key for Gemini

# Use a mock API key if not set, for local testing without actual API calls
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "mock_elevenlabs_key")
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY", "") # Will be provided by Canvas runtime

# --- Global Data Stores (In-memory for simplicity, can be replaced with databases) ---
app = Flask(__name__)
start_time = datetime.datetime.utcnow()

# Simple in-memory user store for demonstration
users: Dict[str, Dict[str, str]] = {}
# Global dict for chat history, keyed by user_id
thread_contexts: defaultdict[str, List[Dict[str, str]]] = defaultdict(list)
# Persistent DB using TinyDB
db_persistent = TinyDB('persistent_data.json')


# --- Utility Functions and Decorators ---

def log_event(event_name: str, data: Dict[str, Any]):
    """Logs events with a timestamp."""
    timestamp = datetime.datetime.utcnow().isoformat()
    log_entry = {"timestamp": timestamp, "event": event_name, "data": data}
    print(json.dumps(log_entry, indent=2))
    # In a real app, this would write to a proper logging system (e.g., ELK stack, CloudWatch)

def error_response(message: str, status_code: int):
    """Generates a standardized JSON error response."""
    log_event("API_Error", {"message": message, "status_code": status_code, "path": request.path})
    return jsonify({"error": message}), status_code

def require_json(f):
    """Decorator to ensure request content type is JSON."""
    @functools.wraps(f)
    def decorated_function(*args, **kwargs):
        if not request.is_json:
            return error_response("Request must be JSON", 400)
        return f(*args, **kwargs)
    return decorated_function

def generate_token():
    """Generates a simple random token for user sessions."""
    return str(uuid.uuid4())

def get_thread_summary(user_id: str, num_messages: int = 5) -> str:
    """Summarizes recent chat history for a given user."""
    history = thread_contexts.get(user_id, [])
    if not history:
        return "No recent chat history."
    recent_messages = history[-num_messages:]
    summary = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_messages])
    return f"Recent chat history summary:\n{summary}"

def update_thread(user_id: str, message: str, role: str = "user"):
    """Adds a message to the user's chat history."""
    thread_contexts[user_id].append({"role": role, "content": message})
    # Optionally trim history to keep it manageable
    max_history_length = 20
    if len(thread_contexts[user_id]) > max_history_length:
        thread_contexts[user_id] = thread_contexts[user_id][-max_history_length:]

# --- AI Core Components (LLM Wrapper, Memory, Agents, Tools) ---

async def ai_completion(messages: List[Dict[str, str]], model: str = "gemini-2.0-flash", temperature: float = 0.7, max_tokens: int = 1024, retry_attempts: int = 3) -> Tuple[str, Dict[str, Any]]:
    """
    Wrapper for making API calls to the Gemini LLM.
    Args:
        messages: A list of message dictionaries (role, content).
        model: The model to use (e.g., "gemini-2.0-flash").
        temperature: Controls randomness.
        max_tokens: Maximum number of tokens in the response.
        retry_attempts: Number of times to retry the API call on failure.
    Returns:
        A tuple of (generated_text, metadata_dict).
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GEMINI_API_KEY}"
    headers = {'Content-Type': 'application/json'}
    payload = {
        "contents": messages,
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens
        }
    }

    log_event("LLM_Call_Attempt", {"model": model, "messages_len": len(messages), "first_msg": messages[0]['content'][:100] if messages else ""})

    for attempt in range(retry_attempts):
        try:
            # Use requests.post here, as `fetch` is a JS-only concept in this context.
            # For a pure Python environment, `requests` is the standard.
            # Note: The original prompt used `fetch` which is for client-side JS.
            # If this Python code were to be run within a Canvas runtime *as JS*,
            # the original `fetch` call would be more appropriate.
            # Assuming this is a standalone Python Flask server:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
            result = response.json()

            if result.get("candidates") and result["candidates"][0].get("content") and result["candidates"][0]["content"].get("parts"):
                generated_text = result["candidates"][0]["content"]["parts"][0]["text"]
                log_event("LLM_Call_Success", {"model": model, "response_len": len(generated_text)})
                return generated_text, result
            else:
                log_event("LLM_Call_Empty_Response", {"model": model, "response": result})
                return "Error: Empty or malformed response from LLM.", result

        except requests.exceptions.RequestException as e:
            log_event("LLM_Call_Error", {"model": model, "attempt": attempt + 1, "error": str(e)})
            if attempt < retry_attempts - 1:
                time.sleep(2 ** attempt) # Exponential backoff
            else:
                raise RuntimeError(f"LLM API call failed after {retry_attempts} attempts: {e}")
        except Exception as e:
            log_event("LLM_Call_Unexpected_Error", {"model": model, "error": str(e)})
            raise RuntimeError(f"An unexpected error occurred during LLM API call: {e}")

class MemoryStore:
    """A simple in-memory key-value store for context. Could be Redis/DB in production."""
    def __init__(self):
        self._store: Dict[str, Dict[str, Any]] = defaultdict(dict)

    def remember(self, user_id: str, key: str, value: Any):
        """Stores a piece of information."""
        self._store[user_id][key] = value
        log_event("Memory_Store_Remember", {"user_id": user_id, "key": key})

    def recall(self, user_id: str, key: str) -> Any | None:
        """Recalls a piece of information."""
        value = self._store[user_id].get(key)
        log_event("Memory_Store_Recall", {"user_id": user_id, "key": key, "found": value is not None})
        return value

    def forget(self, user_id: str, key: str):
        """Removes a piece of information."""
        if key in self._store[user_id]:
            del self._store[user_id][key]
            log_event("Memory_Store_Forget", {"user_id": user_id, "key": key})

datastore = MemoryStore() # Instantiate the MemoryStore

class Tool:
    """Base class for tools."""
    def __init__(self, name: str, description: str, parameters_schema: Dict[str, Any]):
        self.name = name
        self.description = description
        self.parameters_schema = parameters_schema

    def execute(self, **kwargs) -> Any:
        raise NotImplementedError

class GoogleSearchTool(Tool):
    def __init__(self):
        super().__init__(
            name="google_search",
            description="Performs a Google search and returns relevant snippets.",
            parameters_schema={
                "query": {"type": "string", "description": "The search query."}
            }
        )
    def execute(self, query: str) -> List[Dict[str, str]]:
        log_event("Tool_Execute", {"tool": self.name, "query": query})
        # Mock Google Search API call
        mock_results = [
            {"title": f"Result 1 for '{query}'", "snippet": f"This is a snippet about {query} from a reliable source.", "url": f"http://example.com/search1_{query.replace(' ', '_')}"},
            {"title": f"Result 2 for '{query}'", "snippet": f"More details on {query} can be found here.", "url": f"http://example.com/search2_{query.replace(' ', '_')}"}
        ]
        return mock_results

class WebBrowsingTool(Tool):
    def __init__(self):
        super().__init__(
            name="web_browsing",
            description="Browses a given URL and extracts main content.",
            parameters_schema={
                "url": {"type": "string", "description": "The URL to browse."}
            }
        )
    def execute(self, url: str) -> str:
        log_event("Tool_Execute", {"tool": self.name, "url": url})
        # Mock web browsing - in reality, use a library like BeautifulSoup or Playwright
        if "example.com" in url:
            return f"Content of {url}: This is simulated content from a webpage about a specific topic related to {url.split('/')[-1]}."
        return "Could not retrieve content from the specified URL."

class ContentFetcherTool(Tool):
    def __init__(self):
        super().__init__(
            name="content_fetcher",
            description="Fetches content from internal knowledge bases or specified references.",
            parameters_schema={
                "source_references": {"type": "array", "description": "List of source IDs or references to fetch.", "items": {"type": "object", "properties": {"id": {"type": "string"}}}}
            }
        )
    def execute(self, source_references: List[Dict[str, str]]) -> str:
        log_event("Tool_Execute", {"tool": self.name, "source_refs": source_references})
        mock_data = {
            "kb_article_123": "This article explains the basics of quantum computing, including qubits and superposition.",
            "doc_xyz": "Project Thor details: Phase 1 completed, Phase 2 in planning for Q3.",
            "user_manual_v2": "The user manual covers installation, usage, and troubleshooting for product X."
        }
        fetched_content = []
        for ref in source_references:
            content = mock_data.get(ref.get("id"), f"Content for ID {ref.get('id')} not found.")
            fetched_content.append(f"--- Source {ref.get('id')} ---\n{content}\n")
        return "\n".join(fetched_content)

class AlgorandBalanceTool(Tool):
    def __init__(self):
        super().__init__(
            name="algorand_balance",
            description="Retrieves the ALGO balance for a given Algorand wallet address.",
            parameters_schema={
                "wallet_address": {"type": "string", "description": "The Algorand wallet address."}
            }
        )
    def execute(self, wallet_address: str) -> Dict[str, Any]:
        log_event("Tool_Execute", {"tool": self.name, "wallet_address": wallet_address})
        # Mock Algorand API call (in reality, use py-algorand-sdk)
        if not wallet_address.startswith("AAAA"): # Simple mock validation
            return {"Error": "Invalid Algorand wallet address format."}
        
        # Simulate varying balances for different addresses
        import random
        balance = round(random.uniform(10.0, 1000.0), 2)
        return {"balance_algo": balance, "unit": "ALGO", "status": "success"}

class TextToSpeechTool(Tool):
    def __init__(self):
        super().__init__(
            name="text_to_speech",
            description="Converts text to speech using an external service (e.g., ElevenLabs).",
            parameters_schema={
                "text": {"type": "string", "description": "The text to convert to speech."},
                "voice": {"type": "string", "description": "The voice to use (e.g., 'Rachel', 'Adam'). Default is 'Rachel'.", "default": "Rachel"}
            }
        )
    def execute(self, text: str, voice: str = "Rachel") -> bytes:
        log_event("Tool_Execute", {"tool": self.name, "text_len": len(text), "voice": voice})
        # Mock ElevenLabs API call
        if not ELEVENLABS_API_KEY or ELEVENLABS_API_KEY == "mock_elevenlabs_key":
            return b"MOCKED_AUDIO_BYTES_FOR_TEXT_TO_SPEECH"
        
        # In a real scenario, make an actual API call
        # url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        # headers = {
        #     "Accept": "audio/mpeg",
        #     "Content-Type": "application/json",
        #     "xi-api-key": ELEVENLABS_API_KEY
        # }
        # data = {
        #     "text": text,
        #     "model_id": "eleven_monolingual_v1",
        #     "voice_settings": {
        #         "stability": 0.5,
        #         "similarity_boost": 0.75
        #     }
        # }
        # response = requests.post(url, headers=headers, json=data)
        # response.raise_for_status()
        # return response.content
        return b"MOCKED_AUDIO_BYTES_FOR_TEXT_TO_SPEECH"


# Initialize available tools
TOOL_REGISTRY: Dict[str, Tool] = {
    "google_search": GoogleSearchTool(),
    "web_browsing": WebBrowsingTool(),
    "content_fetcher": ContentFetcherTool(),
    "algorand_balance": AlgorandBalanceTool(),
    "text_to_speech": TextToSpeechTool(),
}

class BaseAgent:
    """Base class for all agents."""
    def __init__(self, name: str, description: str, model: str = "gemini-2.0-flash"):
        self.name = name
        self.description = description
        self.model = model

    async def respond(self, prompt: str, chat_history: List[Dict[str, str]] = None) -> str:
        """Generates a response based on a prompt and optional chat history."""
        raise NotImplementedError

class RewriterAgent(BaseAgent):
    """Agent specialized in rewriting and rephrasing text."""
    def __init__(self):
        super().__init__(name="RewriterAgent", description="Specializes in rephrasing, clarifying, and optimizing text and prompts.")

    async def respond(self, text_to_rewrite: str) -> str:
        system_prompt = (
            "You are an expert text rewriter. Your task is to rephrase the given text, "
            "optimizing it for clarity, conciseness, and impact. "
            "Consider the implied intent and make the output more effective. "
            "If the input is a prompt, optimize it for an AI model."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Please rewrite this:\n{text_to_rewrite}"}
        ]
        log_event("RewriterAgent_Call", {"text_len": len(text_to_rewrite)})
        response, _ = await ai_completion(messages, model=self.model)
        log_event("RewriterAgent_Response", {"response_len": len(response)})
        return response

class ResearcherAgent(BaseAgent):
    """Agent specialized in gathering information using tools."""
    def __init__(self, tools: Dict[str, Tool]):
        super().__init__(name="ResearcherAgent", description="Gathers information by utilizing various research tools like web search.")
        self.tools = tools # Researcher agent has access to specific tools

    async def respond(self, query: str) -> str:
        system_prompt = (
            "You are a diligent Researcher Agent. Your goal is to answer the user's query by effectively "
            "using the available tools. Structure your response clearly, citing any information found. "
            "If you need to use a tool, output a JSON object like this: "
            "```json\n{\"tool_name\": \"tool_name\", \"parameters\": {\"param1\": \"value1\"}}\n``` "
            "Then, after the tool output, synthesize your answer. If you have enough information, "
            "provide a comprehensive answer. Available tools: "
            f"{json.dumps({name: tool.description for name, tool in self.tools.items()})}"
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]

        # First, try to get LLM to use a tool or provide a direct answer
        log_event("ResearcherAgent_Thinking", {"query": query})
        tool_or_answer, _ = await ai_completion(messages, model=self.model, temperature=0.2)

        try:
            # Check if the LLM decided to use a tool
            tool_call_data = json.loads(tool_or_answer.strip())
            tool_name = tool_call_data.get("tool_name")
            tool_parameters = tool_call_data.get("parameters", {})

            if tool_name in self.tools:
                log_event("ResearcherAgent_Tool_Call", {"tool_name": tool_name, "parameters": tool_parameters})
                tool_output = self.tools[tool_name].execute(**tool_parameters)
                
                # Now, use the tool output to generate the final response
                messages.append({"role": "assistant", "content": json.dumps(tool_call_data)})
                messages.append({"role": "tool", "content": json.dumps(tool_output)})
                messages.append({"role": "user", "content": "Based on the tool output, please provide a concise answer to my original query."})

                final_answer, _ = await ai_completion(messages, model=self.model)
                return final_answer
            else:
                return f"Researcher Agent Error: Requested unknown tool '{tool_name}'."
        except json.JSONDecodeError:
            # If not a tool call, it's a direct answer from the LLM
            log_event("ResearcherAgent_Direct_Answer", {"response_len": len(tool_or_answer)})
            return tool_or_answer
        except Exception as e:
            log_event("ResearcherAgent_Execution_Error", {"error": str(e)})
            return f"Researcher Agent encountered an error: {e}. Attempted response: {tool_or_answer}"


class PersonalAssistantAgent(BaseAgent):
    """The main orchestrator agent that plans and executes multi-step tasks."""
    def __init__(self):
        super().__init__(name="PersonalAssistantAgent", description="Orchestrates multi-step tasks involving tools and other agents.")
        self.memory = MemoryStore() # Orchestrator has its own memory for planning

    async def plan(self, raw_prompt: str, combined_context: str, available_tools: Dict[str, Tool], orchestrated_agents: Dict[str, BaseAgent]) -> List[Dict[str, Any]]:
        """
        Generates an execution plan using LLM.
        The plan is a list of steps, each with a type (tool_call, agent_call, final_response).
        """
        tool_descriptions = "\n".join([f"- {name}: {tool.description} Parameters: {tool.parameters_schema}" for name, tool in available_tools.items()])
        agent_descriptions = "\n".join([f"- {name}: {agent.description}" for name, agent in orchestrated_agents.items()])

        system_prompt = (
            "You are an expert AI Orchestrator. Your task is to create a multi-step execution plan "
            "to fulfill a user's request. Each step must be a valid JSON object. "
            "The plan should be an array of steps. "
            "Available tools and agents are described below. "
            "After all necessary information is gathered, include a 'final_response' step "
            "to synthesize the answer.\n\n"
            "Here are the available tool specifications:\n"
            f"{tool_descriptions}\n\n"
            "Here are the available agents:\n"
            f"{agent_descriptions}\n\n"
            "Possible step types and their JSON structure:\n"
            "1. Tool Call:\n"
            "   ```json\n"
            "   {\"step_name\": \"unique_name\", \"step_type\": \"tool_call\", \"tool_name\": \"tool_name_here\", \"parameters\": {\"param1\": \"value1\"}}\n"
            "   ```\n"
            "2. Agent Call:\n"
            "   ```json\n"
            "   {\"step_name\": \"unique_name\", \"step_type\": \"agent_call\", \"agent_name\": \"agent_name_here\", \"agent_prompt\": \"Prompt for the agent. Can use {{previous_step_output}} placeholders.\"}\n"
            "   ```\n"
            "3. Final Response:\n"
            "   ```json\n"
            "   {\"step_name\": \"final_answer\", \"step_type\": \"final_response\", \"response_synthesis_prompt\": \"Summarize {{step1_output}}, {{step2_output}}, and answer the original query: {{original_query_placeholder}}.\"}\n"
            "   ```\n"
            "Ensure 'response_synthesis_prompt' for final_response explicitly tells me how to synthesize the response. "
            "Use placeholders `{{step_name_output}}` to refer to outputs of previous steps. "
            "The final_response step should always be the last step. "
            "Your output must be a single JSON array representing the plan."
        )

        user_message = (
            f"User's request: {raw_prompt}\n"
            f"Current context: {combined_context}\n\n"
            "Generate the execution plan as a JSON array:"
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        log_event("Orchestrator_Planning_Call", {"prompt_len": len(raw_prompt)})
        try:
            plan_json_str, _ = await ai_completion(messages, model=self.model, temperature=0.4)
            log_event("Orchestrator_Planning_Response", {"plan_str_len": len(plan_json_str)})
            plan = json.loads(plan_json_str)
            if not isinstance(plan, list):
                raise ValueError("Plan must be a JSON array.")
            return plan
        except (json.JSONDecodeError, ValueError) as e:
            log_event("Orchestrator_Planning_Error", {"error": str(e), "raw_plan_response": plan_json_str})
            raise RuntimeError(f"Failed to generate a valid execution plan: {e}. Raw response: {plan_json_str}")
        except Exception as e:
            log_event("Orchestrator_Planning_Unexpected_Error", {"error": str(e)})
            raise RuntimeError(f"An unexpected error occurred during planning: {e}")

    async def execute_plan(self, user_id: str, execution_plan: List[Dict[str, Any]],
                           orchestrated_agents: Dict[str, BaseAgent], available_tools: Dict[str, Tool],
                           raw_prompt: str) -> Tuple[str, List[Dict[str, Any]]]:
        """Executes the given plan step by step."""
        step_outputs: Dict[str, Any] = {"original_query_placeholder": raw_prompt} # Store outputs of steps
        execution_log: List[Dict[str, Any]] = []
        final_response_prompt = ""

        for step in execution_plan:
            step_name = step.get("step_name", f"step_{uuid.uuid4().hex[:8]}")
            step_type = step.get("step_type")

            log_event("Orchestrator_Executing_Step", {"user_id": user_id, "step_name": step_name, "step_type": step_type})

            try:
                # Replace placeholders in step parameters/prompts
                processed_step = json.loads(json.dumps(step).replace("{{original_query_placeholder}}", raw_prompt))
                for placeholder, value in step_outputs.items():
                    processed_step = json.loads(json.dumps(processed_step).replace(f"{{{{{placeholder}}}}}", json.dumps(value))) # Use json.dumps for potentially complex outputs

                if step_type == "tool_call":
                    tool_name = processed_step.get("tool_name")
                    tool_parameters = processed_step.get("parameters", {})
                    if tool_name in available_tools:
                        tool_output = available_tools[tool_name].execute(**tool_parameters)
                        step_outputs[f"{step_name}_output"] = tool_output
                        execution_log.append({"type": "tool_output", "step_name": step_name, "tool_name": tool_name, "parameters": tool_parameters, "output": tool_output})
                        log_event("Orchestrator_Tool_Success", {"step_name": step_name, "tool_name": tool_name})
                    else:
                        tool_output = f"Error: Tool '{tool_name}' not found."
                        step_outputs[f"{step_name}_output"] = tool_output
                        execution_log.append({"type": "tool_error", "step_name": step_name, "error": tool_output})
                        log_event("Orchestrator_Tool_Error_NotFound", {"step_name": step_name, "tool_name": tool_name})

                elif step_type == "agent_call":
                    agent_name = processed_step.get("agent_name")
                    agent_prompt = processed_step.get("agent_prompt")
                    if agent_name in orchestrated_agents:
                        # Agents are async, so await their response
                        agent_response = await orchestrated_agents[agent_name].respond(agent_prompt)
                        step_outputs[f"{step_name}_output"] = agent_response
                        execution_log.append({"type": "agent_output", "step_name": step_name, "agent_name": agent_name, "prompt": agent_prompt, "output": agent_response})
                        log_event("Orchestrator_Agent_Success", {"step_name": step_name, "agent_name": agent_name})
                    else:
                        agent_response = f"Error: Agent '{agent_name}' not found."
                        step_outputs[f"{step_name}_output"] = agent_response
                        execution_log.append({"type": "agent_error", "step_name": step_name, "error": agent_response})
                        log_event("Orchestrator_Agent_Error_NotFound", {"step_name": step_name, "agent_name": agent_name})

                elif step_type == "final_response":
                    final_response_prompt = processed_step.get("response_synthesis_prompt", "Please provide a final answer based on the information gathered.")
                    log_event("Orchestrator_Final_Response_Planned", {"user_id": user_id, "prompt": final_response_prompt})
                    break # Break loop, as final response step is usually last
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
                # For now, log and continue. Could implement stop-on-error logic.

        # Synthesize final response
        if not final_response_prompt:
            # Fallback if no final_response step was specified in the plan
            final_response_prompt = "No specific final response prompt was provided by the plan. Synthesize a response based on the outputs of the previous steps:\n"
            for k, v in step_outputs.items():
                if k != "original_query_placeholder": # Don't add the original prompt to the synthesis data itself
                    final_response_prompt += f"{k}: {v}\n"
            final_response_prompt += f"Original Query: {raw_prompt}"

        # Replace placeholders in the final response prompt with actual outputs
        # This loop is redundant if `processed_step` already handled all placeholders,
        # but it acts as a safeguard for the final synthesis prompt.
        for placeholder, value in step_outputs.items():
            # Ensure value is stringified for replacement
            final_response_prompt = final_response_prompt.replace(f"{{{{{placeholder}}}}}", str(value))
        
        # New: Reflection step (post-execution analysis)
        reflection_prompt = (
            "Analyze the following execution log and step outputs. Identify any gaps, "
            "inconsistencies, or areas where the initial plan might have failed. "
            "Suggest how the overall process could be improved for this type of query. "
            "This is for internal review, provide a concise summary of findings."
            f"\n\nExecution Log: {json.dumps(execution_log, indent=2)}\n"
            f"Step Outputs: {json.dumps(step_outputs, indent=2)}"
        )
        reflection_messages = [
            {"role": "system", "content": "You are an AI process auditor and improver."},
            {"role": "user", "content": reflection_prompt}
        ]
        
        try:
            reflection_notes, _ = await ai_completion(reflection_messages, model=self.model, temperature=0.3)
            log_event("Orchestrator_Reflection_Completed", {"user_id": user_id, "reflection_len": len(reflection_notes)})
            execution_log.append({"type": "reflection", "notes": reflection_notes})
        except Exception as e:
            log_event("Orchestrator_Reflection_Error", {"user_id": user_id, "error": str(e)})
            execution_log.append({"type": "reflection_error", "error": f"Failed to perform reflection: {e}"})

        try:
            # The system prompt here for the final response can be generic
            final_messages = [
                {"role": "system", "content": "You are a helpful assistant. Consolidate the provided information and answer the user's request comprehensively and clearly. Address all parts of the original request."},
                {"role": "user", "content": final_response_prompt}
            ]
            final_answer, _ = await ai_completion(final_messages, model=self.model)
            log_event("Orchestrator_Final_Response_Generated", {"user_id": user_id, "response_len": len(final_answer)})
            execution_log.append({"type": "final_response", "output": final_answer})
            return final_answer, execution_log
        except Exception as e:
            error_msg = f"Error synthesizing final response: {e}"
            log_event("Orchestrator_Final_Synthesis_Error", {"user_id": user_id, "error": error_msg})
            return f"I managed to get some information, but I couldn't synthesize a complete answer: {error_msg}. Intermediate results: {json.dumps(step_outputs)}", execution_log

    async def plan_and_execute(self, user_id: str, raw_prompt: str, combined_context: str,
                               orchestrated_agents: Dict[str, BaseAgent],
                               available_tools: Dict[str, Tool]) -> Tuple[str, List[Dict[str, Any]]]:
        """Main method to plan and execute a multi-step query."""
        log_event("Orchestrator_Start", {"user_id": user_id, "raw_prompt_len": len(raw_prompt)})
        try:
            # Step 1: Plan
            plan = await self.plan(raw_prompt, combined_context, available_tools, orchestrated_agents)
            log_event("Orchestrator_Plan_Generated", {"user_id": user_id, "plan_steps": len(plan)})
            
            # Step 2: Execute
            final_answer, execution_log = await self.execute_plan(user_id, plan, orchestrated_agents, available_tools, raw_prompt)
            log_event("Orchestrator_Execution_Complete", {"user_id": user_id, "final_answer_len": len(final_answer)})
            return final_answer, execution_log
        except Exception as e:
            error_message = f"Orchestration failed: {e}"
            log_event("Orchestrator_Overall_Failure", {"user_id": user_id, "error": error_message})
            return f"I'm sorry, I encountered an error during the process: {error_message}", [{"type": "orchestration_error", "error": error_message}]


# Instantiate agents that can be orchestrated
rewriter_agent = RewriterAgent()
researcher_agent = ResearcherAgent(tools={"google_search": TOOL_REGISTRY["google_search"], "web_browsing": TOOL_REGISTRY["web_browsing"]})
personal_assistant_agent = PersonalAssistantAgent() # This is the orchestrator

ORCHESTRATED_AGENTS: Dict[str, BaseAgent] = {
    "RewriterAgent": rewriter_agent,
    "ResearcherAgent": researcher_agent,
    "PersonalAssistantAgent": personal_assistant_agent # Self-referential for complex meta-orchestration
}


class PostProcessor:
    """Handles post-processing of responses, e.g., sentiment analysis, keyword tagging."""
    def __init__(self, model: str = "gemini-2.0-flash"):
        self.model = model

    async def tag_keywords(self, text: str) -> List[str]:
        system_prompt = "Extract up to 5 main keywords from the following text. Respond with a comma-separated list of keywords only. No other text."
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ]
        log_event("PostProcessor_Keyword_Tagging", {"text_len": len(text)})
        try:
            keywords_str, _ = await ai_completion(messages, model=self.model, temperature=0.1, max_tokens=100)
            return [k.strip() for k in keywords_str.split(',') if k.strip()]
        except Exception as e:
            log_event("PostProcessor_Keyword_Error", {"error": str(e)})
            return []

    async def sentiment_score(self, text: str) -> str:
        system_prompt = "Analyze the sentiment of the following text. Respond with a single word: 'positive', 'negative', or 'neutral'."
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ]
        log_event("PostProcessor_Sentiment_Analysis", {"text_len": len(text)})
        try:
            sentiment, _ = await ai_completion(messages, model=self.model, temperature=0.1, max_tokens=20)
            return sentiment.strip().lower()
        except Exception as e:
            log_event("PostProcessor_Sentiment_Error", {"error": str(e)})
            return "unknown"

post_processor = PostProcessor()


class Router:
    """Routes prompts to the most appropriate agent using an LLM."""
    def __init__(self, orchestrated_agents: Dict[str, BaseAgent]):
        self.orchestrated_agents = orchestrated_agents
        self.model = "gemini-2.0-flash"

    async def route(self, prompt: str) -> BaseAgent:
        # Provide the router with descriptions of the available agents
        agent_descriptions = "\n".join([f"- {name}: {agent.description}" for name, agent in self.orchestrated_agents.items()])

        system_prompt = (
            "You are an intelligent routing agent. Based on the user's prompt, "
            "determine which of the following agents is best suited to handle the request. "
            "Respond only with the name of the chosen agent (e.g., 'PersonalAssistantAgent', 'RewriterAgent', 'ResearcherAgent').\n\n"
            "Available Agents:\n"
            f"{agent_descriptions}\n\n"
            "If the request requires multiple steps, tool usage, or coordination of information, choose 'PersonalAssistantAgent'. "
            "If it's purely about rephrasing or optimizing text, choose 'RewriterAgent'. "
            "If it primarily involves gathering factual information using external sources, choose 'ResearcherAgent'."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"User's request: {prompt}"}
        ]
        
        log_event("Router_Decision_Call", {"prompt_len": len(prompt)})
        try:
            chosen_agent_name, _ = await ai_completion(messages, model=self.model, temperature=0.1, max_tokens=50)
            chosen_agent_name = chosen_agent_name.strip()
            
            if chosen_agent_name in self.orchestrated_agents:
                log_event("Router_Decision_Success", {"prompt": prompt, "chosen_agent": chosen_agent_name})
                return self.orchestrated_agents[chosen_agent_name]
            else:
                log_event("Router_Decision_Invalid_Agent", {"prompt": prompt, "chosen_agent": chosen_agent_name})
                # Fallback to orchestrator if LLM picks an invalid agent name
                return self.orchestrated_agents["PersonalAssistantAgent"]
        except Exception as e:
            log_event("Router_Decision_Error", {"prompt": prompt, "error": str(e)})
            # Fallback to orchestrator on any error
            return self.orchestrated_agents["PersonalAssistantAgent"]

router = Router(orchestrated_agents=ORCHESTRATED_AGENTS) # Instantiate the router

class SentientChainExecutor:
    """Coordinates the entire AI interaction flow."""
    def __init__(self, memory_store: MemoryStore, agent_router: Router,
                 rewriter_agent: RewriterAgent, post_processor: PostProcessor,
                 thread_contexts_ref: defaultdict, persistent_db: TinyDB,
                 orchestrator_agent: PersonalAssistantAgent,
                 available_tools: Dict[str, Tool],
                 orchestrated_agents: Dict[str, BaseAgent]):
        self.memory_store = memory_store # For simple in-memory K-V pairs
        self.agent_router = agent_router
        self.rewriter_agent = rewriter_agent
        self.post_processor = post_processor
        self.thread_contexts = thread_contexts_ref # Reference to the global dict for chat history
        self.persistent_db = persistent_db # TinyDB instance
        self.orchestrator_agent = orchestrator_agent # The main orchestrator (PersonalAssistantAgent)
        self.available_tools = available_tools # Pass the TOOL_REGISTRY
        self.orchestrated_agents = orchestrated_agents # Pass the list of agents the orchestrator can use

    async def execute_chain(self, user_id: str, prompt: str) -> Tuple[str, Dict[str, Any]]:
        log_event("Executor_Execution_Started", {"user_id": user_id, "prompt_len": len(prompt)})

        # 1. Pre-processing (e.g., prompt enhancement)
        # Use RewriterAgent to enhance the prompt before routing, if desired
        # enhanced_prompt = await self.rewriter_agent.respond(f"Optimize this prompt for an AI: {prompt}")
        # prompt_to_use = enhanced_prompt # For now, let's use the original prompt for clearer flow

        # 2. Routing
        target_agent = await self.agent_router.route(prompt)
        log_event("Executor_Routing_Decision", {"user_id": user_id, "target_agent": target_agent.name})

        # 3. Context Gathering
        chat_history_summary = get_thread_summary(user_id) # Summarize chat history
        user_data = self.memory_store.recall(user_id, "user_profile") # Example of recalling from DataStore
        
        # Example of fetching persistent user data from TinyDB
        persistent_user_records = self.persistent_db.search(Query().user_id == user_id)
        persistent_user_data = persistent_user_records[0] if persistent_user_records else {} # Get first record if exists
        
        combined_context = f"Chat History Summary: {chat_history_summary}\nUser Profile: {user_data}\nPersistent Data: {persistent_user_data}"
        log_event("Executor_Context_Gathered", {"user_id": user_id, "context_len": len(combined_context)})

        raw_response = ""
        execution_log: List[Dict[str, Any]] = [] # To capture detailed steps for debugging/auditing

        try:
            if target_agent == self.orchestrator_agent:
                # Orchestrator handles complex, multi-step queries
                final_answer, log = await self.orchestrator_agent.plan_and_execute(
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
                raw_response = await self.rewriter_agent.respond(prompt)
                execution_log.append({"type": "direct_agent_call", "agent": "RewriterAgent", "response": raw_response})

            elif target_agent == self.orchestrated_agents["ResearcherAgent"]: # Explicitly check for ResearcherAgent
                raw_response = await self.orchestrated_agents["ResearcherAgent"].respond(prompt)
                execution_log.append({"type": "direct_agent_call", "agent": "ResearcherAgent", "response": raw_response})

            else:
                # Fallback for other direct agents if needed, though router should route specifically
                raw_response = await target_agent.respond(prompt, chat_history=[{"role": "user", "content": prompt}]) # Minimal history
                execution_log.append({"type": "direct_agent_call", "agent": target_agent.name, "response": raw_response})

        except Exception as e:
            raw_response = f"An error occurred during agent execution: {e}"
            log_event("Executor_Agent_Execution_Error", {"user_id": user_id, "error": str(e), "target_agent": target_agent.name})
            execution_log.append({"type": "executor_error", "error": str(e)})


        # 4. Post-processing (e.g., sentiment, keywords)
        keywords = await self.post_processor.tag_keywords(raw_response)
        sentiment = await self.post_processor.sentiment_score(raw_response)
        log_event("Executor_Post_Processed", {"user_id": user_id, "keywords": keywords, "sentiment": sentiment})
        execution_log.append({"type": "post_processing", "keywords": keywords, "sentiment": sentiment})

        # 5. Update chat thread
        update_thread(user_id, prompt, role="user")
        update_thread(user_id, raw_response, role="assistant")

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

class TextToSpeechRequest(BaseModel):
    text: str = Field(..., min_length=1)
    voice: str = "Rachel"

class AlgoBalanceRequest(BaseModel):
    wallet_address: str = Field(..., min_length=1)

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
    user_id = str(uuid.uuid4())
    users[username] = {"password": hashed_password, "token": token, "user_id": user_id}
    
    # Store initial user data in persistent DB
    db_persistent.insert({"user_id": user_id, "username": username, "created_at": datetime.datetime.utcnow().isoformat()})
    
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
async def ai_chain_endpoint():
    try:
        req_data = PromptRequest(**request.get_json())
        user_id = req_data.user_id
        user_prompt = req_data.prompt

        # Execute the AI chain
        response_content, metadata = await sentient_executor.execute_chain(user_id, user_prompt)

        return jsonify({
            "response": response_content,
            "metadata": metadata,
            "timestamp": datetime.datetime.utcnow().isoformat()
        }), 200

    except ValidationError as e:
        return error_response(f"Invalid request data: {e.errors()}", 422)
    except RuntimeError as e:
        return error_response(f"AI chain execution failed: {e}", 500)
    except Exception as e:
        log_event("Unhandled_Error_AI_Chain", {"error": str(e), "request_data": request.get_json()})
        return error_response(f"An unexpected error occurred: {e}", 500)

@app.route("/ai/speak", methods=["POST"])
@require_json
async def ai_speak_endpoint():
    try:
        req_data = TextToSpeechRequest(**request.get_json())
        text = req_data.text
        voice = req_data.voice

        audio_bytes = TOOL_REGISTRY["text_to_speech"].execute(text=text, voice=voice)
        # Encode audio bytes to base64 for JSON response
        encoded_audio = base64.b64encode(audio_bytes).decode('utf-8')
        return jsonify({
            "audio_base64": encoded_audio,
            "format": "mp3" # ElevenLabs typically returns mp3
        }), 200
    except ValidationError as e:
        return error_response(f"Invalid request data: {e.errors()}", 422)
    except RuntimeError as e:
        return error_response(str(e), 500)
    except Exception as e:
        log_event("Unhandled_Error_TTS", {"error": str(e), "text_len": len(text)})
        return error_response(f"An unexpected error occurred during speech generation: {e}", 500)

@app.route("/algo/balance", methods=["POST"])
@require_json
def algo_balance_endpoint():
    try:
        req_data = AlgoBalanceRequest(**request.get_json())
        wallet_address = req_data.wallet_address

        # AlgorandBalanceTool already has internal validation
        balance_info = TOOL_REGISTRY["algorand_balance"].execute(wallet_address=wallet_address)
        if "Error" in balance_info:
            return error_response(balance_info["Error"], 500) # Propagate error from tool
        return jsonify({
            "wallet_address": wallet_address,
            "balance": balance_info
        }), 200
    except ValidationError as e:
        return error_response(f"Invalid request data: {e.errors()}", 422)
    except Exception as e:
        log_event("Unhandled_Error_Algo_Balance", {"error": str(e), "wallet_address": wallet_address})
        return error_response(f"An unexpected error occurred during Algorand balance check: {e}", 500)


if __name__ == "__main__":
    # Ensure TinyDB file exists or is created
    # This is a simple file-based DB, fine for demonstration.
    # For production, use a proper database.
    try:
        db_persistent.insert({'_init': True}) # Just to ensure file creation if it doesn't exist
        db_persistent.remove(where('_init').exists())
    except Exception as e:
        print(f"Warning: Could not perform initial TinyDB operation: {e}")

    # Set up a test user for quick API testing if no users exist
    if not users:
        test_username = "testuser"
        test_password = "testpassword"
        hashed_password = bcrypt.hashpw(test_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        test_token = generate_token()
        test_user_id = str(uuid.uuid4())
        users[test_username] = {"password": hashed_password, "token": test_token, "user_id": test_user_id}
        db_persistent.insert({"user_id": test_user_id, "username": test_username, "created_at": datetime.datetime.utcnow().isoformat()})
        print(f"Created a test user: {test_username} with user_id: {test_user_id} and token: {test_token}")
        print("Use this for API calls to /user/login and /ai/chain.")

    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
