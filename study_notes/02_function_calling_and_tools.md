# 02 — Function Calling & Tools

## What is it?
Function calling is a mechanism where the LLM **decides WHEN to call a function and WITH WHAT arguments** — but YOUR code actually executes the function and feeds the result back to the LLM.

---

## Why does it matter?
- LLMs can't do math reliably, access databases, call live APIs, or interact with the real world on their own. Function calling bridges that gap.
- It's the foundational building block of **agents** — an agent is essentially an LLM in a loop that keeps calling tools until the task is done.
- Every real-world AI application (chatbots that book flights, assistants that query databases, product recommendation engines) uses function calling under the hood.

---

## How does it work? (Under the Hood)

### The Core Problem

LLMs are text-in, text-out machines. They can't actually *do* anything:

```
❌ LLMs CANNOT:
   • Do reliable math         (they "guess" 15 × 23 by pattern matching)
   • Query a database         (no network access)
   • Call a REST API           (no HTTP client)
   • Read today's weather      (knowledge cutoff)
   • Execute code              (no runtime)

✅ LLMs CAN:
   • Understand what you want
   • Decide WHICH function to call
   • Figure out WHAT arguments to pass
   • Compose a nice response from function results
```

### The 6-Step Function Calling Loop

This is the most important diagram in the entire course. Every tool-using agent follows this exact loop:

```
Step 1: YOU define functions + their JSON schemas
        ┌─────────────────────────────────────────┐
        │  "calculate" — does math operations      │
        │  "get_weather" — fetches weather data    │
        └─────────────────────────────────────────┘
                          │
                          ▼
Step 2: YOU send user message + schemas to LLM
        ┌─────────────────────────────────────────┐
        │  User: "What's 15 * 23 + 45?"           │
        │  Available functions: [calculate, ...]   │
        └─────────────────────────────────────────┘
                          │
                          ▼
Step 3: LLM returns a FUNCTION CALL (not text!)
        ┌─────────────────────────────────────────┐
        │  I want to call: calculate               │
        │  With args: {operation: "*", a: 15, b:23}│
        └─────────────────────────────────────────┘
                          │
                          ▼
Step 4: YOUR CODE executes the function
        ┌─────────────────────────────────────────┐
        │  result = calculate("*", 15, 23)         │
        │  result = 345                            │
        └─────────────────────────────────────────┘
                          │
                          ▼
Step 5: YOU send the result back to the LLM
        ┌─────────────────────────────────────────┐
        │  function "calculate" returned: 345      │
        └─────────────────────────────────────────┘
                          │
                          ▼
Step 6: LLM generates final human-readable response
        ┌─────────────────────────────────────────┐
        │  "15 × 23 = 345. Then 345 + 45 = 390.   │
        │   So 15 × 23 + 45 = 390."               │
        └─────────────────────────────────────────┘
```

**Critical insight:** The LLM never executes the function. It only *asks* for a function to be called. YOUR code is the executor. The LLM is the brain; your code is the hands.

### What the LLM Actually Sees

When you provide function schemas, the LLM's system prompt effectively becomes:

```
You have access to these functions:

1. calculate(operation, a, b) - Perform basic math operations
   - operation: one of "+", "-", "*", "/"
   - a: first number
   - b: second number

2. get_weather(city, country) - Get weather for a city
   - city: name of the city
   - country: country code (default "US")

If a user's request can be answered by calling one of these functions,
respond with the function name and arguments instead of answering directly.
```

The LLM then decides: "Is this a question I can answer from my training data, or do I need a function?"

---

## OpenAI Function Calling (Raw API)

This is from `2_What_Makes_An_Agent/openai_function_calling_feature.ipynb`.

### Step 1: Define the actual Python function

```python
def calculate(operation: str, a: float, b: float) -> float:
    """
    Perform basic mathematical operations

    Args:
        operation: The operation to perform (+, -, *, /)
        a: First number
        b: Second number

    Returns:
        Result of the calculation
    """
    if operation == "+":
        return a + b
    elif operation == "-":
        return a - b
    elif operation == "*":
        return a * b
    elif operation == "/":
        return a / b if b != 0 else "Error: Division by zero"
    else:
        return "Error: Unsupported operation"
```

### Step 2: Define the JSON schema that tells the LLM about this function

```python
import json

calculator_function = {
    "name": "calculate",
    "description": "Perform basic mathematical operations like addition, subtraction, multiplication, and division",
    "parameters": {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "description": "The mathematical operation to perform",
                "enum": ["+", "-", "*", "/"]
            },
            "a": {
                "type": "number",
                "description": "The first number"
            },
            "b": {
                "type": "number",
                "description": "The second number"
            }
        },
        "required": ["operation", "a", "b"]
    }
}
```

**Key parts of the schema:**
- `name` — what the function is called (must match your Python function)
- `description` — tells the LLM *when* to use this function (this is crucial!)
- `parameters` — JSON Schema describing the arguments
- `enum` — restricts values to a specific set
- `required` — which parameters must be provided

### Step 3: The full function calling conversation

```python
import openai
import json

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def chat_with_calculator(user_message: str):
    messages = [
        {"role": "system", "content": "You are a calculation specialist."},
        {"role": "user", "content": user_message}
    ]

    # First API call — send message + function schemas
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        functions=[calculator_function],
        function_call="auto"  # Let OpenAI decide when to call functions
    )

    message = response.choices[0].message

    # Check if the LLM wants to call a function
    if message.function_call:
        function_name = message.function_call.name
        function_args = json.loads(message.function_call.arguments)

        print(f"🔧 OpenAI wants to call: {function_name}")
        print(f"📝 Arguments: {function_args}")

        # YOUR CODE executes the function
        if function_name == "calculate":
            result = calculate(**function_args)

        # Add the assistant's function call to conversation history
        messages.append({
            "role": "assistant",
            "content": None,
            "function_call": {
                "name": function_name,
                "arguments": json.dumps(function_args)
            }
        })

        # Add the function result to conversation history
        messages.append({
            "role": "function",
            "name": function_name,
            "content": str(result)
        })

        # Second API call — LLM generates final response using the result
        final_response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages
        )
        return final_response.choices[0].message.content
```

### Step 4: Test it

```python
result = chat_with_calculator("What's 15 * 23 + 45? Calculate step by step.")
# Output:
# 🔧 OpenAI wants to call: calculate
# 📝 Arguments: {'operation': '*', 'a': 15, 'b': 23}
# "15 × 23 = 345. Then 345 + 45 = 390."
```

### What happens when no function is needed?

```python
result = chat_with_calculator("Who is the PM of India?")
# Output: None (no function call triggered — LLM answers directly)
```

The LLM looked at the available functions (just `calculate`), decided none of them help with "Who is the PM of India?", and answered from its training data instead. This is `function_call="auto"` in action.

---

## Multiple Functions: The FunctionHandler Pattern

From `2_What_Makes_An_Agent/openai_function_calling_feature.ipynb` — handling multiple tools:

```python
class FunctionHandler:
    def __init__(self):
        # Registry: map function names to actual Python functions
        self.functions = {
            "calculate": calculate,
            "get_weather": get_weather
        }
        # Schemas: tell the LLM what's available
        self.function_schemas = [
            calculator_function,
            weather_function
        ]

    def call_function(self, function_name: str, arguments: dict):
        """Call the appropriate function with given arguments"""
        if function_name in self.functions:
            return self.functions[function_name](**arguments)
        else:
            return f"Function {function_name} not found"

    def chat(self, user_message: str):
        """Enhanced chat with multiple function support"""
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

            print(f"🔧 Calling function: {function_name}")
            print(f"📝 Arguments: {function_args}")

            # Execute the function
            result = self.call_function(function_name, function_args)

            # Continue conversation with function result
            messages.extend([
                {
                    "role": "assistant",
                    "content": None,
                    "function_call": {
                        "name": function_name,
                        "arguments": json.dumps(function_args)
                    }
                },
                {
                    "role": "function",
                    "name": function_name,
                    "content": str(result)
                }
            ])

            final_response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages
            )

            return final_response.choices[0].message.content

        return message.content

# Usage
handler = FunctionHandler()
handler.chat("What is the temperature in Tokyo?")
# 🔧 Calling function: get_weather
# 📝 Arguments: {'city': 'Tokyo', 'country': 'JP'}
# "The current temperature in Tokyo is 28°C, with rainy conditions..."

handler.chat("What is the result of 2+2?")
# 🔧 Calling function: calculate
# 📝 Arguments: {'operation': '+', 'a': 2, 'b': 2}
# "The result of 2 + 2 is 4."
```

**Notice how the LLM automatically routes to the right function.** It sees two function schemas, reads the descriptions, and picks the one that matches the user's intent. Weather question → `get_weather`. Math question → `calculate`.

---

## LangChain's Tool Abstraction

LangChain simplifies the OpenAI pattern with its `Tool` class (from `3_Multi_AgenticAI_Frameworks/multi_agent_frameworks_intro_class_work.ipynb`):

### Using `Tool` class

```python
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

def safe_calculate(expression: str) -> str:
    """
    Calculate simple arithmetic operations.
    Args:
        expression: Mathematical statement as a string
    Returns:
        The final result as a string
    """
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {str(e)}"

def search_corpus(query: str) -> str:
    """
    Look up documents for topics related to agentic/multi-agentic AI.
    Args:
        query: Topic name (not descriptive)
    Returns:
        Text/information for the queried topic
    """
    corpus = {
        "langgraph": "Langgraph is used for stateful workflows, retries, and HITL",
        "crewai": "CrewAI focused on role-based multi-agent collaboration",
        "autogen": "Autogen enables multi-agent conversations"
    }
    query = query.lower()
    for key, value in corpus.items():
        if key in query:
            return value
    return "No relevant results found"

# Wrap Python functions as LangChain Tools
tools = [
    Tool(
        name="calculator",
        func=safe_calculate,
        description="Calculate simple arithmetic operations"
    ),
    Tool(
        name="search",
        func=search_corpus,
        description="Look up documents for topics related to agentic/multi-agentic AI, pass the topic name only"
    )
]

# Create an agent with these tools
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
agent = create_agent(model=model, tools=tools)
```

### Running the LangChain agent

```python
response = agent.invoke({
    "messages": [
        {"role": "system", "content": "Use only tools to answer queries. If the tool can't respond, say 'I don't know'"},
        {"role": "user", "content": "What is 8% of 20000?"}
    ]
})

print(response["messages"][-1].content)
# "8% of 20000 is 1600."
```

What happened behind the scenes:
```
User: "What is 8% of 20000?"
  → LLM calls: calculator("20000 * 0.08")
  → Tool returns: "1600.0"
  → LLM says: "8% of 20000 is 1600."
```

### The agent correctly refuses when tools can't help

```python
response = agent.invoke({
    "messages": [
        {"role": "system", "content": "Use only tools to answer. If the tool can't respond, say 'I don't know'"},
        {"role": "user", "content": "Who is the PM of India?"}
    ]
})

print(response["messages"][-1].content)
# "I don't know."
```

The agent looked at its tools (calculator and search_corpus), realized neither can answer "Who is the PM of India?", and followed the system prompt instruction to say "I don't know."

---

## OpenAI Function Calling vs LangChain Tools — Comparison

```
                     OpenAI Raw API              LangChain Tool
─────────────────    ──────────────────          ──────────────────
Schema definition    Manual JSON dict            Automatic from docstring
                                                 (or manual via Tool class)

Function execution   You write the loop          Agent handles the loop
                     (check → execute → feed)    automatically

Multi-tool routing   You write if/elif           Agent handles routing

Conversation mgmt    You manage messages list    Agent manages state

Best for             Learning how it works       Building real applications
                     Fine-grained control        Rapid prototyping
```

### The @tool Decorator (Simplest Way)

LangChain also supports a `@tool` decorator for even simpler tool definition:

```python
from langchain_core.tools import tool

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers together."""
    return a * b

# LangChain auto-generates the JSON schema from:
# - function name → tool name
# - type hints → parameter types
# - docstring → description
```

This is equivalent to writing the full JSON schema manually, but with much less code.

---

## How does it connect to other topics?

- **LLM Fundamentals (See: `01_genai_and_llm_fundamentals.md`)** — Function calling solves the LLM limitations discussed there (can't do math, no live data access, knowledge cutoff).
- **Agents (See: `05_what_makes_an_agent.md`)** — An agent = LLM + tools + reasoning loop. Function calling is how the LLM uses tools. `create_agent()` wraps the entire function calling loop into a single invoke.
- **Multi-Agent Systems (See: `06_multi_agent_frameworks.md`)** — In multi-agent setups (like the product recommendation project), each agent has its own set of tools. The Discovery Agent calls search tools, the Review Agent calls sentiment tools, etc.
- **Product Recommendation Project (See: `09_building_product_recommendation_agent.md`)** — The 4-agent system (Discovery → Spec → Review → Recommendation) is built entirely on top of tool-calling agents.

---

## Code Examples

### Minimal function calling example (raw OpenAI)

```python
import openai, json, os
from dotenv import load_dotenv

load_dotenv(override=True)
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 1. Define function
def add(a, b):
    return a + b

# 2. Define schema
schema = {
    "name": "add",
    "description": "Add two numbers",
    "parameters": {
        "type": "object",
        "properties": {
            "a": {"type": "number"},
            "b": {"type": "number"}
        },
        "required": ["a", "b"]
    }
}

# 3. Call LLM with schema
resp = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "What is 7 + 3?"}],
    functions=[schema],
    function_call="auto"
)

# 4. Check if function was called
msg = resp.choices[0].message
if msg.function_call:
    args = json.loads(msg.function_call.arguments)
    result = add(**args)
    print(f"Function returned: {result}")  # 10
```

### Weather tool with mock data (from the repo)

```python
def get_weather(city: str, country: str = "US") -> str:
    """Get current weather for a city"""
    mock_weather_data = {
        "new york": {"temp": 22, "condition": "sunny", "humidity": 60},
        "london": {"temp": 15, "condition": "cloudy", "humidity": 80},
        "tokyo": {"temp": 28, "condition": "rainy", "humidity": 75}
    }

    city_key = city.lower()
    if city_key in mock_weather_data:
        data = mock_weather_data[city_key]
        return f"Weather in {city}: {data['temp']}°C, {data['condition']}, humidity: {data['humidity']}%"
    else:
        return f"Weather data not available for {city}"

weather_function = {
    "name": "get_weather",
    "description": "Get current weather information for a specific city",
    "parameters": {
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "Name of the city"
            },
            "country": {
                "type": "string",
                "description": "Country code (e.g., US, UK, JP)",
                "default": "US"
            }
        },
        "required": ["city"]
    }
}
```

---

## Common Mistakes

1. **Vague function descriptions**
   ```python
   # ❌ LLM won't know when to use this
   "description": "Does stuff with numbers"

   # ✅ Clear about what and when
   "description": "Perform basic mathematical operations like addition, subtraction, multiplication, and division"
   ```
   The description is the LLM's only guide for deciding when to call your function. Be specific.

2. **Forgetting to send the function result back to the LLM**
   The loop has TWO API calls: one to get the function call, one to get the final answer. If you skip the second call, you just have a raw number (345) instead of a human-friendly response ("15 × 23 = 345").

3. **Not handling the "no function call" case**
   ```python
   # ❌ Crashes when LLM answers directly without calling a function
   args = json.loads(message.function_call.arguments)

   # ✅ Check first
   if message.function_call:
       args = json.loads(message.function_call.arguments)
       # ... handle function call
   else:
       return message.content  # LLM answered directly
   ```

4. **Mismatched function name in schema vs Python**
   If your schema says `"name": "calculate"` but your Python function is called `calc()`, the dispatch will fail silently.

5. **Using `eval()` without sanitization**
   The repo's `safe_calculate` uses `eval()` for simplicity, but in production this is a **security vulnerability**. Never `eval()` user input in real applications.

6. **Not including `required` fields in the schema**
   If you omit `"required": ["operation", "a", "b"]`, the LLM might skip arguments and your function will crash with missing parameters.

7. **Thinking the LLM executes the function**
   The LLM NEVER runs your code. It only outputs: "I'd like to call function X with arguments Y." Your code does the actual execution. This is a fundamental misunderstanding that leads to security and architecture mistakes.

---

## Practice Exercises

1. **Build a Unit Converter:** Create a function calling setup with a `convert_units(value, from_unit, to_unit)` function. It should handle km↔miles, kg↔lbs, and °C↔°F. Write the JSON schema, then test with prompts like "How many miles is 10 kilometers?" and "Convert 100°F to Celsius."

2. **Add Error Handling:** Take the `chat_with_calculator` function from the repo and add proper error handling for: (a) division by zero, (b) the LLM passing invalid arguments, and (c) network timeouts on the API call.

3. **Multi-Tool Agent:** Create a LangChain agent with THREE tools: a calculator, a dictionary lookup (mock data), and a date/time tool (using Python's `datetime`). Test it with mixed queries like "What day is it today and what does 'ephemeral' mean?"

4. **Trace the Loop:** For the FunctionHandler example, add print statements at every step of the 6-step loop. Call `handler.chat("What is 100 / 4?")` and verify you can see all 6 steps happening. Write down what the `messages` list looks like after each step.

5. **Schema Design Challenge:** Write JSON schemas for these real-world functions (don't implement the functions, just the schemas): (a) `search_products(query, category, max_price, sort_by)`, (b) `send_email(to, subject, body, cc)`, (c) `book_meeting(title, date, time, attendees, duration_minutes)`. Think carefully about `enum` values, `required` fields, and descriptions.
