# 01 — GenAI & LLM Fundamentals

## What is it?
Generative AI (GenAI) is a branch of artificial intelligence that **creates new content** — text, images, code, audio — rather than just classifying or predicting labels on existing data.

---

## Why does it matter?
- GenAI powers tools like ChatGPT, GitHub Copilot, and DALL·E that are transforming every industry.
- Understanding how LLMs work under the hood lets you **prompt better**, **debug faster**, and **build real AI agents** instead of just calling APIs blindly.
- Every agentic AI system (multi-agent pipelines, self-reflection loops, tool-calling agents) is built on top of LLM fundamentals — if this foundation is shaky, everything on top breaks.

---

## How does it work? (Under the Hood)

### GenAI vs Traditional AI — The Key Difference

```
Traditional AI (Discriminative)          Generative AI
─────────────────────────────────        ─────────────────────────────
Input: image of a cat                    Input: "Draw me a cat"
Output: label → "cat" (classification)   Output: brand-new image of a cat

Input: email text                        Input: "Write a marketing email"
Output: "spam" or "not spam"             Output: full email text, never seen before
```

Traditional AI learns **boundaries between categories**. GenAI learns **the distribution of the data itself** so it can generate new samples from that distribution.

### What is an LLM (Large Language Model)?

An LLM is a specific type of GenAI model trained on massive text data to **predict the next token** in a sequence. That's it — the entire magic is next-token prediction done really, really well.

```
Prompt:  "The capital of France is"
                                    ↓
         LLM predicts next token → "Paris"
                                    ↓
         Then predicts next     → "."
                                    ↓
         Then predicts next     → [STOP]
```

The model generates text **one token at a time**, feeding its own output back as input for the next step. This is called **autoregressive generation**.

### Transformer Architecture (Simplified)

Transformers are the neural network architecture behind all modern LLMs (GPT, Llama, Mistral, etc.).

```
                    ┌──────────────────────┐
  Input Text   ──► │  Tokenizer           │ ──► Tokens (numbers)
                    └──────────────────────┘
                              │
                              ▼
                    ┌──────────────────────┐
                    │  Embedding Layer     │  Maps each token to a vector
                    └──────────────────────┘
                              │
                              ▼
                    ┌──────────────────────┐
                    │  Positional Encoding │  Adds position info (word order)
                    └──────────────────────┘
                              │
                              ▼
                  ┌────────────────────────────┐
                  │  Transformer Block (×N)     │
                  │  ┌──────────────────────┐   │
                  │  │ Self-Attention        │   │ ◄── THE key innovation
                  │  │ "Which other words    │   │
                  │  │  should I focus on?"  │   │
                  │  └──────────────────────┘   │
                  │  ┌──────────────────────┐   │
                  │  │ Feed-Forward Network  │   │
                  │  └──────────────────────┘   │
                  └────────────────────────────┘
                              │
                              ▼
                    ┌──────────────────────┐
                    │  Output Layer        │ ──► Probability of EVERY possible
                    └──────────────────────┘     next token in vocabulary
```

**Self-Attention** is the breakthrough: for each word, the model calculates how much "attention" it should pay to every other word in the input. This lets it handle long-range dependencies like:

> "The **cat** that sat on the mat in the living room of the house on the hill **was** sleeping."

The model links "was" back to "cat" (singular → singular verb), even across many words.

### Key Concepts You Must Know

#### 1. Tokens
Tokens are the chunks an LLM reads — not exactly words, not exactly characters.

```
"Hello world"        → ["Hello", " world"]           (2 tokens)
"Tokenization"       → ["Token", "ization"]           (2 tokens)
"ChatGPT is amazing" → ["Chat", "G", "PT", " is", " amazing"]  (5 tokens)
```

**Why it matters:** You pay per token. Context windows are measured in tokens. A rough rule: **1 token ≈ 4 characters** or **¾ of a word** in English.

#### 2. Temperature
Controls randomness in the model's output.

```
Temperature = 0.0  →  Always picks the most likely token (deterministic)
Temperature = 0.7  →  Some creativity, good default for general use
Temperature = 1.0  →  Very creative, might go off-topic
Temperature = 2.0  →  Practically random nonsense
```

Think of it like a "creativity dial":
- **Low temperature (0–0.3):** Factual answers, code generation, math
- **Medium temperature (0.5–0.7):** Conversations, explanations
- **High temperature (0.8–1.0+):** Creative writing, brainstorming

#### 3. Context Window
The maximum number of tokens the model can "see" at once (input + output combined).

```
GPT-4o         → 128,000 tokens  (~96,000 words)
GPT-4o-mini    → 128,000 tokens  (cheaper, faster)
Llama 3.1 8B   →   8,192 tokens  (smaller but fast via Groq)
```

If your conversation exceeds the context window, the model "forgets" the earliest messages.

#### 4. Model Variants Used in This Course

```
Model              Provider     Speed    Cost     Best For
─────────────────  ──────────   ─────    ─────    ─────────────────────
gpt-4o             OpenAI       Fast     Medium   General tasks, agents
gpt-4o-mini        OpenAI       Faster   Cheap    Simple tasks, high volume
llama-3.1-8b       Groq         V.Fast   Free*    Quick experiments
Mistral models     HuggingFace  Varies   Free*    Self-hosted inference
```

*Free tier or open-source with usage limits.

---

## How LangChain Wraps LLMs

LangChain provides a **unified interface** so you can swap LLM providers without changing your code logic.

### Setting up with OpenAI (from `1_Introduction/intro.ipynb`)

```python
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load API keys from .env file
load_dotenv(override=True)

# Initialize the LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",   # pick your model
    temperature=0.7         # creativity dial
)

# Call the LLM — this is the fundamental operation
response = llm.invoke("Hi, who is the PM of India?")
print(response.content)
# Output: "The Prime Minister of India is **Narendra Modi**."
```

**Key point:** `invoke()` returns an `AIMessage` object, not a plain string. Access the text via `.content`.

### The AIMessage Object

When you call `llm.invoke(...)`, you get back a rich object:

```python
AIMessage(
    content='The Prime Minister of India is **Narendra Modi**.',
    response_metadata={
        'token_usage': {
            'completion_tokens': 35,
            'prompt_tokens': 15,
            'total_tokens': 50
        },
        'model_name': 'gpt-4o-mini',
        'finish_reason': 'stop'
    },
    usage_metadata={
        'input_tokens': 15,
        'output_tokens': 35,
        'total_tokens': 50
    }
)
```

Useful fields:
- `response.content` → the actual text answer
- `response.usage_metadata` → how many tokens you used (cost tracking!)
- `response.response_metadata['finish_reason']` → why the model stopped (`'stop'` = natural end, `'length'` = hit token limit)

### Setting up with Groq / Llama (from `1_Introduction/intro.ipynb`)

```python
from langchain_groq import ChatGroq

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.7
)

response = llm.invoke("Explain diffusion models simply")
print(response.content)
```

### Setting up with OpenRouter (from `1_Introduction/intro.ipynb`)

```python
from langchain_openrouter import ChatOpenRouter

model = ChatOpenRouter(
    model="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"
)

response = model.invoke("What is the capital of France?")
print(response.content)
# Output: "The capital of France is Paris."
```

**Notice the pattern:** Every provider uses the same `.invoke()` method. That's the power of LangChain's abstraction.

---

## API Keys and `.env` Files — Why and How

### Why not hardcode keys?

```python
# ❌ NEVER DO THIS — key ends up in Git history
client = openai.OpenAI(api_key="sk-abc123...")

# ✅ DO THIS — key stays in .env file (which is .gitignored)
from dotenv import load_dotenv
import os

load_dotenv(override=True)
api_key = os.getenv("OPENAI_API_KEY")
```

### The `.env` file format

```
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxxxxxx
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxx
OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxxxxxx
```

### How `load_dotenv()` works

```
Your Project/
├── .env              ← contains OPENAI_API_KEY=sk-...
├── .gitignore         ← contains ".env" so it's never committed
├── intro.ipynb
└── requirements.txt
```

`load_dotenv(override=True)` reads `.env` and injects values into `os.environ`. The `override=True` ensures `.env` values overwrite any existing system env vars (useful when switching between keys).

---

## Limitations of LLMs

### 1. Hallucinations
LLMs confidently generate **plausible-sounding but incorrect** information.

```
Q: "What paper did Dr. Smith publish in Nature in 2019?"
A: "Dr. Smith published 'Quantum Entanglement in Biological Systems'
    in Nature, Vol 573, pp 234-241."
    
    ^^^ This paper, author, and citation may be completely fabricated.
```

### 2. Non-determinism
Same prompt, different answers each time (unless temperature=0).

```python
# Run 1: "The PM of India is Narendra Modi."
# Run 2: "India's Prime Minister is Narendra Modi, who has been in office since 2014."
# Run 3: "Narendra Modi serves as the PM of India."
```

This is why the repo shows slightly different outputs in the notebook cells — it's expected behavior.

### 3. Knowledge Cutoff
LLMs only know what was in their training data. They don't know about events after their cutoff date unless you give them tools (See: **02_function_calling_and_tools.md**).

### 4. Can't Do Real Computation
LLMs "fake" math by pattern-matching, not calculating. For `15 * 23`, the LLM doesn't multiply — it predicts what tokens typically follow that pattern. This is exactly why **function calling** exists (See: **02_function_calling_and_tools.md**).

---

## How does it connect to other topics?

- **Prompting (See: `03_prompt_engineering.md`)** — How you write your input dramatically changes the LLM's output. Zero-shot, few-shot, and instruction prompting are all techniques to get better results from the same model.
- **Function Calling (See: `02_function_calling_and_tools.md`)** — The solution to LLM limitations. Can't do math? Give it a calculator tool. Can't access live data? Give it an API tool.
- **Structured Output (See: `03_prompt_engineering.md`)** — Getting the LLM to return JSON or specific formats instead of free-form text.
- **Agents (See: `05_what_makes_an_agent.md`)** — When you combine an LLM with tools and a reasoning loop, you get an agent.

---

## Code Examples

### Minimal "Hello LLM" script

```python
# hello_llm.py — The simplest possible LLM call
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv(override=True)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
response = llm.invoke("What is 2 + 2?")
print(response.content)       # "2 + 2 equals 4."
print(response.usage_metadata) # {'input_tokens': 14, 'output_tokens': 9, ...}
```

### Comparing temperature settings

```python
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

prompt = "Write a one-sentence summary of Python."

for temp in [0.0, 0.5, 1.0]:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=temp)
    result = llm.invoke(prompt)
    print(f"temp={temp}: {result.content}\n")
```

### Switching between providers

```python
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv(override=True)

# Same question, two different providers
question = "Explain what a neural network is in one sentence."

openai_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
groq_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

print("OpenAI:", openai_llm.invoke(question).content)
print("Groq:  ", groq_llm.invoke(question).content)
```

---

## Common Mistakes

1. **Forgetting `load_dotenv()` before using the API**
   You'll get an authentication error. Always call it at the top of your script.

2. **Printing the `AIMessage` object instead of `.content`**
   ```python
   # ❌ Prints the whole object with metadata
   print(llm.invoke("Hi"))

   # ✅ Prints just the text
   print(llm.invoke("Hi").content)
   ```

3. **Using high temperature for factual tasks**
   Temperature 0.7+ for math or data extraction = hallucinations. Use `temperature=0` for anything requiring accuracy.

4. **Hardcoding API keys in notebooks**
   Even "just for testing." Keys get committed to Git, pushed to GitHub, and scraped by bots within minutes.

5. **Not understanding token limits**
   Stuffing a 50-page PDF into a prompt for a model with 8K context window will silently truncate your input — the model just won't see the end of your document.

6. **Assuming LLMs are deterministic**
   Even with temperature=0, minor backend changes can cause slightly different outputs. Don't write tests that assert exact string matches against LLM output.

---

## Practice Exercises

1. **Token Counter:** Write a script that sends 5 different prompts to `gpt-4o-mini` and prints the `input_tokens`, `output_tokens`, and `total_tokens` for each. Which prompt used the most tokens? Why?

2. **Temperature Experiment:** Send the exact same creative prompt ("Write a haiku about Python programming") to the same model 3 times each at temperature 0.0, 0.5, and 1.0. Record the outputs. At what temperature do you start seeing different results across runs?

3. **Provider Swap:** Take the Groq/Llama code from `1_Introduction/intro.ipynb` and modify it to use OpenAI's `gpt-4o-mini` instead. Then modify it to use OpenRouter. Verify all three give reasonable answers to the same question. What differences do you notice?

4. **Break the Context Window:** Use `ChatGroq` with `llama-3.1-8b-instant` (8K context). Try sending a very long prompt (copy-paste a long article). What error do you get? How would you handle this in a real application?

5. **Hallucination Detector:** Ask the LLM a very specific factual question you know the answer to (e.g., "What was the GDP of India in Q3 2025?"). Compare the LLM's answer with a verified source. Document where the LLM gets facts wrong or makes things up.
