# 8 — Practice Playground

> **Goal:** Build one capstone project that forces you to use **every concept** you learned in this course. No notebook cells yet — this is your blueprint. Read it, understand it, then build it yourself cell by cell.

---

## What You Will Build

**Project name:** `Personal Research Assistant`

A single AI system that can:
1. Accept a research question from you (e.g., "What are the best laptops for AI under ₹1,50,000?")
2. Search the web for current information
3. Load relevant web pages / PDFs into a vector database
4. Retrieve the most useful chunks
5. Use structured output to extract clean facts
6. Call tools when needed (calculator, unit converter, date)
7. Produce a final, cited report

This one project practices:
- LLM basics (`ChatOpenAI`, `ChatGroq`)
- Prompt engineering (system/human messages, few-shot)
- Pydantic structured output
- Function calling / tools
- Agent pattern (LLM + tools + reasoning loop)
- RAG (load → chunk → embed → store → retrieve → generate)
- Vector databases (FAISS)
- Multi-step orchestration (pure Python)

---

## Demo Blueprint

Below is the complete architecture. Do **not** copy-paste it blindly. Read each block, understand why it exists, then create your own notebook and write the cells yourself.

### Step 1: Environment Setup

```python
from dotenv import load_dotenv
import os

load_dotenv(override=True)

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY", "")
```

**Why:** Your keys live in `.env`. Your code reads them so you never hardcode secrets.

---

### Step 2: Choose Your LLM

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.3
)
```

**Why:** `temperature=0.3` keeps answers focused but not robotic. `gpt-4o-mini` is cheap and fast.

---

### Step 3: Define a Search Tool

```python
from langchain_community.tools import DuckDuckGoSearchResults

search_tool = DuckDuckGoSearchResults(max_results=5)
```

**Why:** LLMs cannot browse the internet. A search tool gives them eyes.

---

### Step 4: Build a Simple Tool-Using Agent

```python
from langchain_core.tools import Tool
from langchain.agents import create_agent

# Define actual Python functions
def calculator(expression: str) -> str:
    """Evaluate a simple math expression."""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"

def today() -> str:
    """Return today's date."""
    from datetime import date
    return str(date.today())

# Wrap them as LangChain Tools
tools = [
    Tool(name="calculator", func=calculator, description="Use for math"),
    Tool(name="today", func=today, description="Use for today's date"),
    Tool(name="web_search", func=search_tool.run, description="Use for current information")
]

# Create the agent
agent = create_agent(model=llm, tools=tools)
```

**Why:** The agent sees the tool names + descriptions and decides which one to call based on your question.

---

### Step 5: Test the Agent

```python
response = agent.invoke({
    "messages": [
        {"role": "user", "content": "What is 15% of 24000 and what is trending in AI today?"}
    ]
})
print(response["messages"][-1].content)
```

**Why:** The agent should call `calculator` for the math and `web_search` for the AI trend. This is the **ReAct loop** in action.

---

### Step 6: Add Structured Output with Pydantic

```python
from pydantic import BaseModel, Field
from typing import List

class ResearchFact(BaseModel):
    claim: str = Field(description="A single factual statement")
    source: str = Field(description="URL or source of the claim")
    confidence: str = Field(description="High, Medium, or Low")

class ResearchReport(BaseModel):
    question: str = Field(description="Original research question")
    summary: str = Field(description="Short summary of findings")
    facts: List[ResearchFact] = Field(description="List of extracted facts")

report_chain = (
    prompt
    | llm.with_structured_output(ResearchReport)
)
```

**Why:** Instead of free text, the LLM returns a typed object you can reliably process.

---

### Step 7: Add RAG

```python
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 1. Load a web page
loader = WebBaseLoader("https://example.com/article")
docs = loader.load()

# 2. Chunk it
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# 3. Embed and store
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(chunks, embeddings)

# 4. Retrieve
retrieved = vector_store.similarity_search("best laptop for AI", k=3)
```

**Why:** RAG grounds the LLM in external documents instead of relying on training memory.

---

### Step 8: Put It All Together

```python
def research_assistant(question: str) -> ResearchReport:
    # 1. Search the web
    search_results = search_tool.run(question)

    # 2. Load top result pages into vector store
    # (Optional: parse URLs from search_results)

    # 3. Retrieve relevant chunks
    context = vector_store.similarity_search(question, k=3)
    context_text = "\n\n".join(c.page_content for c in context)

    # 4. Generate structured report
    return report_chain.invoke({
        "question": question,
        "context": context_text
    })
```

**Why:** This is the full pipeline. Search → chunk → embed → retrieve → generate structured output.

---

## How to Build It Yourself

1. **Create a new notebook** inside `8_Practice/`, e.g., `my_research_assistant.ipynb`
2. **Write one cell at a time.** Run it. Fix errors before moving on.
3. **Do not copy-paste the blueprint.** Type it yourself so your fingers learn it.
4. **Experiment:** change queries, models, chunk sizes, k values, tools.
5. **Add more tools:** currency converter, translator, weather API.
6. **Add a second agent:** a "Fact Checker" agent that critiques the report.

---

# Under the Hood: How Everything Really Works

This section explains the mechanics behind the code you wrote above.

## 1. How LLM Inference Works

An LLM is a giant probability machine.

```
Input: "The capital of France is"
Model predicts next token probabilities:
  "Paris"    → 0.92
  "London"   → 0.03
  "Berlin"   → 0.01
  ...

It samples one token (Paris), appends it, and repeats.
```

**Key insight:** Every response is built one token at a time by predicting what word likely comes next.

- **Temperature = 0:** always pick the highest probability token → deterministic.
- **Temperature = 1:** sample from the full distribution → creative and varied.

## 2. How Function Calling Works

The LLM never runs your code. It only **asks** to run it.

```
You give the LLM:
  - User question
  - Tool descriptions (name, parameters, when to use)

LLM decides:
  "I need the calculator. Call it with {'expression': '15 * 23'}"

Your code runs the calculator and returns 345.

LLM uses 345 to write the final answer.
```

**Critical point:** The LLM outputs JSON describing the function call. Your Python code executes it. This is why the LLM cannot delete files or make real API calls on its own — your code is in control.

## 3. How Embeddings Work

Embeddings turn meaning into geometry.

```
"King"   → [0.2, -0.5, 0.8, ...]  (384 numbers)
"Queen"  → [0.3, -0.4, 0.7, ...]
"Apple"  → [-0.8, 0.2, 0.1, ...]

"King" and "Queen" are close in vector space.
"Apple" is far away.
```

The embedding model was trained so that texts with similar meanings get similar vectors. This lets us search by meaning instead of keywords.

## 4. How Vector Search Works

```
1. Convert query to vector
2. Compare query vector to every stored vector
3. Return the closest ones

Closest can mean:
  - Cosine similarity: same direction
  - L2 distance: close points
```

FAISS makes this fast by pre-organizing vectors. `IndexFlatL2` is exact but slow for millions. `IndexHNSW` and `IndexIVF` trade a little accuracy for much faster search.

## 5. How RAG Works

RAG = Retrieval + Generation.

```
User asks question
       │
       ▼
  Search / load documents
       │
       ▼
  Split into chunks
       │
       ▼
  Embed chunks → store in vector DB
       │
       ▼
  Embed query → find top-k chunks
       │
       ▼
  Add chunks to prompt
       │
       ▼
  LLM generates answer using chunks
```

**Why it beats raw LLM:**
- Current information (not limited by training cutoff)
- Grounded answers (can cite sources)
- Private data (your documents)

## 6. How Agents Work

An agent is just an LLM in a loop.

```
LOOP:
  1. LLM reads the prompt + available tools
  2. LLM decides: call a tool OR give final answer
  3. If tool call → execute tool → return result
  4. LLM sees result and decides again
  5. Repeat until final answer
```

This pattern is called **ReAct** (Reason + Act). The system prompt tells the LLM how to behave, and the tool descriptions tell it what actions are possible.

## 7. How Structured Output Works

```
You define a Pydantic schema.
LangChain converts it to JSON Schema.
The JSON Schema is sent to the LLM as part of the prompt.
The LLM is constrained to output JSON matching that schema.
LangChain validates the JSON and returns a Pydantic object.
```

Without this, the LLM returns free text. With it, you get reliable, typed data you can feed into the next step.

## 8. How Prompt Engineering Works

A prompt is not magic. It is just previous tokens.

```
P(next_token | prompt)
```

A better prompt shifts the probability distribution toward the tokens you want. That is why:
- Adding examples improves format
- Adding a role improves tone
- Adding output format instructions improves structure

## 9. How Multi-Agent Systems Work

Instead of one agent doing everything, you create specialists.

```
Research Agent  → finds facts
Writer Agent    → turns facts into prose
Critic Agent    → reviews and gives feedback
Editor Agent    → finalizes
```

Each agent has its own system prompt and tools. Data flows between them as structured objects. This is cleaner than one giant prompt.

---

## Suggested Practice Order

Start small. Master each piece before combining them.

```
1. LLM hello world
2. Prompt templates
3. Few-shot prompting
4. Pydantic structured output
5. Function calling (1 tool)
6. Function calling (3 tools)
7. Simple agent
8. Embeddings + cosine similarity
9. FAISS similarity search
10. RAG pipeline (one document)
11. RAG pipeline (web search + multiple docs)
12. Personal Research Assistant (capstone)
```

---

## Remember

- **Read the error.** 90% of bugs are import errors, missing API keys, or wrong variable names.
- **Run cells one by one.** Do not run the whole notebook until you understand each part.
- **Change one thing at a time.** If you change 5 things and it breaks, you won't know which caused it.
- **Build your own version.** The demo above is a starting point. Make it yours.

When you are ready, create your first notebook in this folder and start with Step 1.
