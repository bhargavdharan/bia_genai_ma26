# 05 — What Makes an AI Agent?

## What is it?
An **AI agent** is an LLM that can **think**, **decide**, and **act** — not just generate text, but actually DO things in the real world using tools.

---

## Why does it matter?
Most real-world problems can't be solved by a single text response. You need an AI that can search the web, query databases, run calculations, and combine results — all on its own. Agents are what turn LLMs from fancy autocomplete into actual problem-solving assistants. Every major AI product (ChatGPT plugins, GitHub Copilot, customer support bots) is built on agent architecture.

---

## How does it work? (Under the Hood)

### The Core Formula

```
Agent = LLM + Tools + Memory + Reasoning Loop
```

Each piece plays a role:
- **LLM** — the brain that understands language and makes decisions
- **Tools** — functions the agent can call (calculator, search, API, database)
- **Memory** — conversation history so the agent remembers what happened
- **Reasoning Loop** — the logic that decides: "Should I call a tool or give a final answer?"

### Chatbot vs Agent — The Key Difference

```
CHATBOT (passive):                    AGENT (active):
┌──────────────┐                     ┌──────────────┐
│  You ask     │                     │  You give    │
│  a question  │                     │  a GOAL      │
│              │                     │              │
│  It answers  │                     │  It PLANS    │
│  (one shot)  │                     │  steps       │
│              │                     │              │
│  Done.       │                     │  USES tools  │
│              │                     │              │
└──────────────┘                     │  CHECKS      │
                                     │  results     │
                                     │              │
                                     │  Gives FINAL │
                                     │  answer      │
                                     └──────────────┘
```

A chatbot is like texting a smart friend. An agent is like hiring an employee — you give it a task, and it figures out how to complete it.

### The ReAct Pattern: Reason → Act → Observe → Repeat

ReAct (Reasoning + Acting) is the most common agent pattern. Here's how it works:

```
User: "What's 8% of 20000 and is that more than the average rent in NYC?"

LOOP START
  ┌─────────────────────────────────────────┐
  │ Step 1: REASON (Think)                  │
  │ "I need to calculate 8% of 20000 first" │
  └────────────────┬────────────────────────┘
                   │
  ┌────────────────▼────────────────────────┐
  │ Step 2: ACT (Call Tool)                 │
  │ calculator("20000 * 0.08") → 1600.0    │
  └────────────────┬────────────────────────┘
                   │
  ┌────────────────▼────────────────────────┐
  │ Step 3: OBSERVE (Read Result)           │
  │ "Got 1600. Now I need NYC rent data"    │
  └────────────────┬────────────────────────┘
                   │
  ┌────────────────▼────────────────────────┐
  │ Step 4: REASON again                    │
  │ "I should search for NYC rent info"     │
  └────────────────┬────────────────────────┘
                   │
  ┌────────────────▼────────────────────────┐
  │ Step 5: ACT (Call Tool)                 │
  │ search("average rent NYC 2024")         │
  └────────────────┬────────────────────────┘
                   │
  ┌────────────────▼────────────────────────┐
  │ Step 6: OBSERVE + FINAL ANSWER          │
  │ "1600 < 3500 avg rent. So no."          │
  └─────────────────────────────────────────┘
LOOP END
```

The key insight: the LLM decides **when** to use tools and **which** tool to use. It's not a fixed script — it's dynamic reasoning.

### The Agent Loop (What Actually Happens in Code)

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│ User sends  │────▶│ LLM receives │────▶│ LLM decides: │
│ a message   │     │ message +    │     │ tool_call or │
│             │     │ tool schemas │     │ text answer? │
└─────────────┘     └──────────────┘     └──────┬───────┘
                                                │
                              ┌─────────────────┴─────────────────┐
                              │                                   │
                    ┌─────────▼─────────┐             ┌──────────▼──────────┐
                    │ TEXT ANSWER        │             │ TOOL CALL           │
                    │ → Return to user  │             │ → Execute function  │
                    │ → Done!           │             │ → Get result        │
                    └───────────────────┘             │ → Send back to LLM │
                                                      │ → LLM thinks again │
                                                      └──────────┬─────────┘
                                                                 │
                                                      (loops back to "LLM decides")
```

### How Function Calling Powers Agents

Function calling (See: `02_function_calling_and_tools.md`) is the **mechanism** that makes agents possible. Here's the flow:

1. You define a tool as a Python function + JSON schema
2. The LLM sees the schema and decides when to call it
3. The LLM outputs structured JSON with the function name + arguments
4. Your code executes the function and sends the result back
5. The LLM reads the result and either calls another tool or gives a final answer

This is NOT the LLM running code. The LLM just says "I want to call function X with arguments Y." YOUR code actually runs the function.

---

## How does it connect to other topics?

- **See: `02_function_calling_and_tools.md`** — Function calling is the mechanism that enables tool use. Without it, agents can't interact with external systems.
- **See: `06_multi_agent_frameworks.md`** — When one agent isn't enough, you create multiple specialized agents that collaborate.
- **See: `09_building_product_recommendation_agent.md`** — The mini project sprint is a real 4-agent system where each agent is a specialist (Scout, Engineer, Critic, Advisor).
- **See: `03_prompt_engineering.md`** — The system message that defines an agent's personality and constraints is prompt engineering in action.

---

## Code Examples

### Example 1: How function calling works (from the repo)

This is from `2_What_Makes_An_Agent/openai_function_calling_feature.ipynb`. The LLM can call a calculator:

```python
import openai
import json

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Step 1: Define the actual function
def calculate(operation: str, a: float, b: float) -> float:
    if operation == "+": return a + b
    elif operation == "-": return a - b
    elif operation == "*": return a * b
    elif operation == "/": return a / b if b != 0 else "Error: Division by zero"

# Step 2: Define the schema (tells LLM what the function does)
calculator_function = {
    "name": "calculate",
    "description": "Perform basic mathematical operations",
    "parameters": {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "description": "The mathematical operation to perform",
                "enum": ["+", "-", "*", "/"]
            },
            "a": {"type": "number", "description": "The first number"},
            "b": {"type": "number", "description": "The second number"}
        },
        "required": ["operation", "a", "b"]
    }
}

# Step 3: Send message + function schema to LLM
def chat_with_calculator(user_message: str):
    messages = [
        {"role": "system", "content": "You are a calculation specialist."},
        {"role": "user", "content": user_message}
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        functions=[calculator_function],
        function_call="auto"  # Let LLM decide when to call
    )

    message = response.choices[0].message
    if message.function_call:
        # Step 4: LLM wants to call a function — execute it
        function_name = message.function_call.name
        function_args = json.loads(message.function_call.arguments)
        result = calculate(**function_args)

        # Step 5: Send result back to LLM for final answer
        messages.append({"role": "assistant", "content": None,
                         "function_call": {"name": function_name,
                                           "arguments": json.dumps(function_args)}})
        messages.append({"role": "function", "name": function_name,
                         "content": str(result)})

        final_response = client.chat.completions.create(
            model="gpt-4o", messages=messages
        )
        return final_response.choices[0].message.content
```

When you run `chat_with_calculator("What's 15 * 23 + 45?")`:
```
🔧 OpenAI wants to call: calculate
📝 Arguments: {'operation': '*', 'a': 15, 'b': 23}
→ Result: "First, 15 × 23 = 345. Then 345 + 45 = 390."
```

### Example 2: Multi-function agent handler (from the repo)

This `FunctionHandler` class manages multiple tools — the agent decides which one to use:

```python
class FunctionHandler:
    def __init__(self):
        self.functions = {
            "calculate": calculate,
            "get_weather": get_weather
        }
        self.function_schemas = [calculator_function, weather_function]

    def call_function(self, function_name: str, arguments: dict):
        if function_name in self.functions:
            return self.functions[function_name](**arguments)
        return f"Function {function_name} not found"

    def chat(self, user_message: str):
        messages = [{"role": "user", "content": user_message}]
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            functions=self.function_schemas,
            function_call="auto"
        )
        message = response.choices[0].message

        if message.function_call:
            function_name = message.function_call.name
            function_args = json.loads(message.function_call.arguments)
            result = self.call_function(function_name, function_args)

            # Send result back to LLM
            messages.extend([
                {"role": "assistant", "content": None,
                 "function_call": {"name": function_name,
                                   "arguments": json.dumps(function_args)}},
                {"role": "function", "name": function_name,
                 "content": str(result)}
            ])
            final_response = client.chat.completions.create(
                model="gpt-4o", messages=messages
            )
            return final_response.choices[0].message.content
        return message.content

handler = FunctionHandler()
handler.chat("What is the temperature in Tokyo?")
# → Calls get_weather, returns "28°C, rainy, humidity: 75%"
handler.chat("What is 2+2?")
# → Calls calculate, returns "The result of 2 + 2 is 4."
```

### Example 3: LangChain agent with tools (from the repo)

This is from `3_Multi_AgenticAI_Frameworks/multi_agent_frameworks_intro_class_work.ipynb`:

```python
from langchain.agents import create_agent
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI

# Define tools
def safe_calculate(expression: str) -> str:
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {str(e)}"

def search_corpus(query: str) -> str:
    corpus = {
        "langgraph": "LangGraph is used for stateful workflows, retries, and HITL",
        "crewai": "CrewAI focused on role-based multi-agent collaboration",
        "autogen": "Autogen enables multi-agent conversations"
    }
    query = query.lower()
    for key, value in corpus.items():
        if key in query:
            return value
    return "No relevant results found"

# Wrap functions as LangChain Tools
tools = [
    Tool(name="calculator", func=safe_calculate,
         description="Use for mathematical calculations"),
    Tool(name="search", func=search_corpus,
         description="Search local knowledge corpus for AI framework info")
]

# Create the agent
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
agent = create_agent(model=model, tools=tools)

# Run the agent
response = agent.invoke({
    "messages": [
        {"role": "system", "content": "Use only tools to answer. If no tool can help, say 'I don't know'"},
        {"role": "user", "content": "What is 8% of 20000?"}
    ]
})

print(response["messages"][-1].content)
# → "8% of 20000 is 1600."
```

The agent automatically:
1. Recognized this is a math problem
2. Called the `calculator` tool with `"20000 * 0.08"`
3. Got the result `1600.0`
4. Formatted a nice answer

---

## Common Mistakes

1. **Thinking the LLM runs code** — It doesn't! The LLM outputs "I want to call calculate(*, 15, 23)" as structured JSON. YOUR Python code actually runs `calculate("*", 15, 23)`. The LLM is just a decision-maker.

2. **Forgetting to send tool results back** — After executing a function, you MUST send the result back to the LLM. Otherwise it can't use the result in its response. This is the "function" role message.

3. **Poor tool descriptions** — If your tool description is vague, the LLM won't know when to use it. Be specific: "Use this to perform mathematical calculations" is better than "a useful tool."

4. **Not handling the "no tool needed" case** — Sometimes the LLM can answer directly without tools. Your code must check `if message.function_call:` and handle the else case (just return the text).

5. **Giving the agent too many tools** — Each tool adds tokens to the prompt. If you give an agent 50 tools, it gets confused and slow. Keep it focused — 3-7 tools per agent is a good range.

6. **Not using a system message** — Without a system message, the agent has no personality or constraints. Always define what the agent should (and shouldn't) do.

---

## Practice Exercises

1. **Build a two-tool agent**: Create an agent with a `calculator` tool and a `string_length` tool. Ask it "How many characters are in the word 'hello' multiplied by 3?" — it should use both tools.

2. **Add error handling**: Modify the `FunctionHandler` from Example 2 to handle cases where the LLM calls a function with wrong argument types (e.g., passing a string where a number is expected).

3. **Build a weather + math agent**: Create an agent that can get weather data AND do temperature conversions (Celsius to Fahrenheit). Ask: "What's the temperature in Tokyo in Fahrenheit?"

4. **System message experiment**: Run the same agent with three different system messages:
   - "You are a helpful assistant"
   - "You are a strict calculator that only answers math questions"
   - "You are a sarcastic assistant"
   Observe how the system message changes behavior, even with the same tools.

5. **Trace the agent loop**: Using the LangChain agent from Example 3, print every message in the `response["messages"]` list. Identify which messages are from the user, which are tool calls, which are tool results, and which are the final answer. Draw the full flow on paper.
