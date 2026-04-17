"""

### AI Agent Reasoning Layer ###

# Purpose : Provide single, clean AIAgent class that can support LLM provider using a unified interface.

# Key Responsibilities : Call the LLM service provide via API Key via interface
    
"""


###----------------------------------------------------------------------------------------------------------------###
# Import all required libraries

import importlib
import traceback

###----------------------------------------------------------------------------------------------------------------###
# AI_Providers List (Match from config.py)

PROVIDER_OPENAI   = "OpenAI"
PROVIDER_CLAUDE   = "Anthropic (Claude)"
PROVIDER_GROQ     = "Groq"
PROVIDER_GEMINI   = "Google Gemini"

###----------------------------------------------------------------------------------------------------------------###
# AI_Agent (defined by class)

class AIAgent:   
    def __init__(self, config: dict):
        """
        Initialise the agent with a configuration dictionary.

        Parameters
        ----------
        config : dict
            provider    (str)  — one of the PROVIDER_* constants above
            api_key     (str)  — provider API key (never logged or stored in files)
            model       (str)  — model name string
            max_tokens  (int)  — maximum tokens for each completion
            temperature (float)— sampling temperature (0 = deterministic, 1 = creative)
        """
        self.provider    = config.get("provider",    PROVIDER_OPENAI)
        self.api_key     = config.get("api_key",     "")
        self.model       = config.get("model",       "gpt-3.5-turbo")
        self.max_tokens  = int(config.get("max_tokens",  1000))
        self.temperature = float(config.get("temperature", 0.3))

        # System prompt is injected from config.py via app.py
        self.system_prompt = config.get("system_prompt", "You are a helpful financial reconciliation assistant.")

    #Test Connection (Check the connection status)
    def test_connection(self) -> tuple:        
        if not self.api_key:
            return False, "API key is empty. Please enter your key and save."

        try:            
            test_messages = [{"role": "user", "content": "ping"}]
            reply = self._dispatch(test_messages)
            return True, f" Connection established — model: {self.model}"
        except Exception as exc:
            return False, f" Connection failed: {str(exc)}"

    # Chat (Public)
    def chat(self, messages: list, context: str = "") -> str:        
        if not self.api_key:
            return " No API key configured. Please set your API key in the configuration panel."

        try:
            # Prepend the reconciliation context as a system-level user message
            # This ensures the model always has the data context for every question.
            enriched_messages = []
            if context:
                enriched_messages.append({
                    "role"   : "user",
                    "content": f"Here is the current reconciliation context:\n\n{context}"
                })
                enriched_messages.append({
                    "role"   : "assistant",
                    "content": "Understood. I have reviewed the reconciliation results. How can I help you?"
                })
            enriched_messages.extend(messages)

            return self._dispatch(enriched_messages)

        except Exception as exc:
            return f" LLM error: {str(exc)}\n\nTrace: {traceback.format_exc()}"


    # Routing to right provider
    def _dispatch(self, messages: list) -> str:
        """Route the call to the correct provider implementation."""
        if self.provider == PROVIDER_OPENAI:
            return self._call_openai(messages)
        elif self.provider == PROVIDER_CLAUDE:
            return self._call_anthropic(messages)
        elif self.provider == PROVIDER_GROQ:
            return self._call_groq(messages)
        elif self.provider == PROVIDER_GEMINI:
            return self._call_gemini(messages)
        else:
            return f"Unknown provider: {self.provider}"


    # 1. OPENAI
    def _call_openai(self, messages: list) -> str:
        """Call OpenAI Chat Completions API."""
        openai = self._require_package("openai", "pip install openai")

        client = openai.OpenAI(api_key=self.api_key)

        # Build the full message list with system prompt at top
        full_messages = [{"role": "system", "content": self.system_prompt}] + messages

        response = client.chat.completions.create(
            model       = self.model,
            messages    = full_messages,
            max_tokens  = self.max_tokens,
            temperature = self.temperature,
        )

        return response.choices[0].message.content.strip()

    # 2. Anthropic Claude

    def _call_anthropic(self, messages: list) -> str:
        """Call Anthropic Messages API."""
        anthropic = self._require_package("anthropic", "pip install anthropic")

        client = anthropic.Anthropic(api_key=self.api_key)

        # Anthropic uses a separate 'system' parameter (not in messages list)
        response = client.messages.create(
            model      = self.model,
            max_tokens = self.max_tokens,
            system     = self.system_prompt,
            messages   = messages,
        )

        # Response content is a list of blocks; extract text block
        for block in response.content:
            if block.type == "text":
                return block.text.strip()

        return "No response received from Claude."


    # 3.Grok 
    def _call_groq(self, messages: list) -> str:
        """Call Groq Chat Completions API (OpenAI-compatible)."""
        groq = self._require_package("groq", "pip install groq")

        client = groq.Groq(api_key=self.api_key)

        full_messages = [{"role": "system", "content": self.system_prompt}] + messages

        response = client.chat.completions.create(
            model       = self.model,
            messages    = full_messages,
            max_tokens  = self.max_tokens,
            temperature = self.temperature,
        )

        return response.choices[0].message.content.strip()


    # 4. Google Gemini
    def _call_gemini(self, messages: list) -> str:
        """Call Google Gemini GenerativeAI API."""
        genai = self._require_package("google.generativeai", "pip install google-generativeai")

        genai.configure(api_key=self.api_key)

        model = genai.GenerativeModel(
            model_name   = self.model,
            system_instruction = self.system_prompt,
        )

        # Gemini uses its own message format: {role: "user"|"model", parts: [text]}
        history     = []
        last_user   = None

        for msg in messages:
            role    = "user" if msg["role"] == "user" else "model"
            content = msg["content"]

            if role == "user":
                if last_user:
                    history.append({"role": "user",  "parts": [last_user]})
                    history.append({"role": "model", "parts": ["Understood."]})
                last_user = content
            else:
                if last_user:
                    history.append({"role": "user",  "parts": [last_user]})
                    last_user = None
                history.append({"role": "model", "parts": [content]})

        if not last_user:
            return "No user message to send."

        chat_session = model.start_chat(history=history)
        response     = chat_session.send_message(last_user)

        return response.text.strip()


    # Ensure specific LLM package is installed
    def _require_package(package_name: str, install_cmd: str):
        
        try:
            return importlib.import_module(package_name)
        except ModuleNotFoundError:
            raise RuntimeError(
                f"Package '{package_name}' is not installed. "
                f"Run:  {install_cmd}"
            )


# Quick Connection test

if __name__ == "__main__":
    import sys
    print("AI Agent — Quick Connection Test")
    provider = input("Provider (OpenAI / Anthropic (Claude) / Groq / Google Gemini): ").strip()
    api_key  = input("API Key: ").strip()
    model    = input("Model name: ").strip()

    agent = AIAgent({
        "provider"   : provider,
        "api_key"    : api_key,
        "model"      : model,
        "max_tokens" : 100,
        "temperature": 0.3,
    })

    ok, msg = agent.test_connection()
    print(f"\nResult: {msg}")

    if ok:
        reply = agent.chat(
            messages=[{"role": "user", "content": "Say hello in 10 words or less."}],
            context=""
        )
        print(f"Reply : {reply}")


