# Product Recommendation Agent System

## What is this project?

This is a **multi-agent AI system** that helps users find the best product to buy. Instead of manually visiting 10+ websites, reading reviews, and comparing specs — this system does it all automatically using AI agents.

Think of it like having 4 expert assistants working for you:

```
You: "I want the best laptop for AI development under ₹1,50,000"

Agent 0 (Scout)      → Searches the internet, finds 5 relevant laptops
Agent 1 (Engineer)   → Looks up detailed specs (CPU, GPU, RAM, etc.) for each
Agent 2 (Critic)     → Reads reviews, finds pros/cons, checks complaints
Agent 3 (Advisor)    → Scores everything, ranks products, gives final advice
```

---

## How does it work?

### The Pipeline (Step by Step)

```
User Query
    │
    ▼
┌─────────────────────────────────┐
│  AGENT 0: Discovery Agent       │
│                                 │
│  • Takes your query             │
│  • Searches the web             │
│  • Finds 5 relevant products    │
│  • Returns: brand, model, why   │
│                                 │
│  Tool: Tavily / DuckDuckGo      │
│  Output: DiscoveryOutput        │
└───────────────┬─────────────────┘
                │ (list of products)
                ▼
┌─────────────────────────────────┐
│  AGENT 1: Specification Agent   │
│                                 │
│  • Takes product list           │
│  • Searches specs for each      │
│  • Extracts: CPU, GPU, RAM,     │
│    Storage, Display, Battery,   │
│    Weight, Price                 │
│                                 │
│  Tool: Tavily / DuckDuckGo      │
│  Output: SpecificationOutput    │
└───────────────┬─────────────────┘
                │ (specs for all products)
                ▼
┌─────────────────────────────────┐
│  AGENT 2: Review Analysis Agent │
│                                 │
│  • Takes product list           │
│  • Searches reviews for each    │
│  • Analyzes: sentiment, pros,   │
│    cons, common issues, expert  │
│    opinions                     │
│                                 │
│  Tool: Tavily / DuckDuckGo      │
│  Output: ReviewOutput           │
└───────────────┬─────────────────┘
                │ (review analysis)
                ▼
┌─────────────────────────────────┐
│  AGENT 3: Recommendation Agent  │
│                                 │
│  • Takes ALL data from above    │
│  • Scores each product (0-10):  │
│    - Performance (30%)          │
│    - Value for Money (25%)      │
│    - User Reviews (20%)         │
│    - AI Readiness (25%)         │
│  • Ranks products               │
│  • Gives final verdict          │
│                                 │
│  Tool: LLM reasoning only       │
│  Output: RecommendationOutput   │
└───────────────┬─────────────────┘
                │
                ▼
        Final Advisory Report
        (Top Pick + Rankings + Verdict)
```

---

## Key Concepts Used

### 1. Pydantic (Structured Output)
**Problem**: When you ask an LLM a question, it returns free-form text. Every time the format is different.

**Solution**: Pydantic models define a strict schema. The LLM is forced to return data in that exact format.

```python
# Without Pydantic — unpredictable format every time:
"ASUS ROG Strix is a great laptop..."
{"laptop": "ASUS ROG Strix"}
["ASUS", "Lenovo"]

# With Pydantic — always the same structure:
{"brand": "ASUS", "model": "ROG Strix G16", "reason": "Best GPU for AI"}
```

### 2. LangChain Chains (Prompt | LLM)
A chain connects a **prompt template** to an **LLM** using the pipe (`|`) operator:

```python
chain = prompt | llm.with_structured_output(MySchema)
result = chain.invoke({"key": "value"})  # Returns MySchema object
```

### 3. Search Tools
- **Tavily API**: High-quality AI-optimized search (needs API key, 1000 free/month)
- **DuckDuckGo**: Free search, no API key needed (less reliable results)

### 4. Custom Orchestration (No Frameworks)
This project does NOT use LangGraph, CrewAI, or AutoGen. The agents are orchestrated using plain Python — one function calls the next, passing data through.

---

## Project Structure

```
mini_project_sprint/
│
├── README.md                              ← You are here
├── .env                                   ← API keys (OPENAI_API_KEY, TAVILY_API_KEY)
│
├── productrecomdagent.ipynb               ← Original notebook (experimentation, drafts)
├── product_recommendation_agent.ipynb     ← Clean version using Tavily search
└── product_recommendation_agent_ddg.ipynb ← Clean version using DuckDuckGo search
```

### Which notebook should I use?

| Notebook | Search | API Key Needed | Best For |
|----------|--------|---------------|----------|
| `product_recommendation_agent.ipynb` | Tavily | OPENAI + TAVILY | Better search quality |
| `product_recommendation_agent_ddg.ipynb` | DuckDuckGo | OPENAI only | No extra API key |

---

## Setup Instructions

### Step 1: Install Python packages

```bash
pip install langchain langchain-openai langchain-community pydantic python-dotenv tavily-python duckduckgo-search
```

### Step 2: Create `.env` file

Create a `.env` file in the project folder with:

```
OPENAI_API_KEY=your-openai-api-key-here
TAVILY_API_KEY=your-tavily-api-key-here
```

- Get OpenAI key: https://platform.openai.com/api-keys
- Get Tavily key: https://tavily.com (free tier: 1000 searches/month)

### Step 3: Run the notebook

Open either notebook in Jupyter/VS Code and run cells top to bottom.

---

## Notebook Learning Sections

Both notebooks are divided into these sections for learning:

| Section | What You Learn |
|---------|---------------|
| Part 1: Setup | How to load API keys, initialize LLM |
| Part 2: Understanding Pydantic | What structured output is and why it matters |
| Part 3: Understanding Search Tools | How web search works in code |
| Part 4: Agent 0 - Discovery | How to build a prompt + chain + invoke |
| Part 5: Agent 1 - Specifications | How to pass data between agents |
| Part 6: Agent 2 - Review Analysis | How to analyze sentiment with LLM |
| Part 7: Agent 3 - Recommendation | How to score and rank with LLM |
| Part 8: Full Pipeline | How to orchestrate all agents together |

Each section has:
- **Explanation** of what the code does and why
- **Code cells** with inline comments
- **Experiment cells** where you can try things yourself

---

## Example Output

```
🏆 TOP PICK: ASUS TUF Gaming A15
   Best GPU performance for AI workloads within budget

📊 RANKINGS:
  #1 ASUS TUF Gaming A15 — Overall: 8.5/10
  #2 Lenovo Legion 5 — Overall: 8.2/10
  #3 Acer Predator Helios Neo 16 — Overall: 7.9/10
  #4 HP Omen 16 — Overall: 7.5/10
  #5 MSI Katana 15 — Overall: 7.2/10

⚖️  TRADE-OFFS: ASUS has better GPU but Lenovo has better thermals...

✅ VERDICT: Go with ASUS TUF Gaming A15 for the best AI development
   experience. If cooling is a priority, consider Lenovo Legion 5.

🎯 CONFIDENCE: Medium
```

---

## Technologies

- **Python 3.13**
- **Pydantic v2** — Data validation and structured output
- **LangChain** — LLM orchestration framework
- **OpenAI GPT** — Large Language Model
- **Tavily** — AI-optimized web search API
- **DuckDuckGo** — Free web search (no API key)
