# 00 — Agentic AI Learning Roadmap

## What is this?

Your complete learning path from "What is an LLM?" to "Build a full multi-agent product recommendation system." These notes cover every concept in the BIA GenAI course, organized so each topic builds on the previous one.

---

## The Learning Path

```
┌─────────────────────────────────────────────────────────────┐
│                    YOUR LEARNING JOURNEY                     │
└─────────────────────────────────────────────────────────────┘

  01  GenAI & LLMs (Foundation)
   │   What are LLMs? How do they generate text?
   │   You learn: tokens, temperature, API calls, LangChain setup
   │
   ▼
  02  Function Calling & Tools (LLM gets superpowers)
   │   How do LLMs call external functions?
   │   You learn: OpenAI function calling, tool schemas, JSON args
   │
   ▼
  03  Prompt Engineering (How to talk to LLMs)
   │   Zero-shot, few-shot, instruction prompting
   │   You learn: prompt templates, system/human messages, Chain of Thought
   │
   ▼
  04  Pydantic & Structured Output (Make LLM output reliable)
   │   Force LLMs to return clean, typed data every time
   │   You learn: BaseModel, Field, with_structured_output(), validation
   │
   ▼
  05  What Makes an Agent (LLM + Tools + Reasoning)
   │   The anatomy of an AI agent
   │   You learn: agent = LLM + tools + prompt, ReAct pattern, tool binding
   │
   ▼
  06  Multi-Agent Frameworks (Teams of agents)
   │   LangChain agents, CrewAI, custom orchestration
   │   You learn: agent teams, task delegation, sequential/parallel pipelines
   │
   ▼
  07  Self-Reflection & Critique (Agents that improve themselves)
   │   Generate → Critique → Refine loops
   │   You learn: rubric evaluation, reflection memory, Constitutional AI
   │
   ▼
  08  Sentiment Analysis (Understanding opinions)
   │   How LLMs analyze reviews, opinions, emotions
   │   You learn: rule-based vs ML vs LLM sentiment, aspect-based analysis
   │
   ▼
  10  RAG Fundamentals (Ground LLMs in real data)
   │   Load, chunk, embed, store, retrieve, generate
   │   You learn: document loaders, text splitters, embeddings, FAISS
   │
   ▼
  11  Vector Databases & FAISS (Semantic search)
   │   Store and search text by meaning instead of keywords
   │   You learn: embeddings, cosine similarity, FAISS indexes
   │
   ▼
  09  Product Recommendation Agent (Everything together!)
       The capstone: 4-agent system that recommends products
       You build: Discovery → Specs → Reviews → Recommendation pipeline
```

---

## Note-by-Note Guide

### 01 — GenAI & LLMs (`01_genai_and_llm_fundamentals.md`)

**One-line summary**: LLMs are neural networks that predict the next token — and that simple idea powers everything in this course.

**What you learn**:
- What Generative AI and LLMs are (transformers, tokens, attention)
- How to call OpenAI and Groq APIs via LangChain
- Temperature, model selection, `ChatOpenAI` setup
- The difference between completion and chat models

**Prerequisites**: None — this is where you start

**Repo folder**: `intro.ipynb`, `1_Introduction/`

---

### 02 — Function Calling & Tools (`02_function_calling_and_tools.md`)

**One-line summary**: Function calling lets LLMs decide WHEN to use external tools and WHAT arguments to pass — turning a chatbot into an agent.

**What you learn**:
- OpenAI's function calling mechanism (tool schemas, JSON arguments)
- How the LLM decides when to call a function vs answer directly
- The request/response cycle: LLM → function call → result → LLM
- Building your own tool definitions

**Prerequisites**: 01 (need to understand LLM basics)

**Repo folder**: `2_What_Makes_An_Agent/`

---

### 03 — Prompt Engineering (`03_prompt_engineering.md`)

**One-line summary**: The right prompt turns a generic LLM into a specialist — zero-shot, few-shot, and instruction prompting are your main tools.

**What you learn**:
- Zero-shot prompting (just ask)
- Few-shot prompting (give examples)
- Instruction prompting (detailed system messages)
- ChatPromptTemplate and variable substitution
- Chain of Thought (CoT) reasoning

**Prerequisites**: 01 (need to know how to call LLMs)

**Repo folder**: `4_Prompt_Essentials/`

---

### 04 — Pydantic & Structured Output (`04_pydantic_and_structured_output.md`)

**One-line summary**: Pydantic schemas force the LLM to return data in an exact format — making outputs reliable and machine-readable.

**What you learn**:
- `BaseModel`, `Field(description=...)`, type hints
- `llm.with_structured_output(Schema)` in LangChain
- Why structured output is essential for multi-agent systems
- Validation and error handling

**Prerequisites**: 01, 03 (need LLM + prompt basics)

**Repo folder**: `mini_project_sprint/productrecomdagent.ipynb` (Pydantic intro section)

---

### 05 — What Makes an Agent (`05_what_makes_an_agent.md`)

**One-line summary**: An agent = LLM + Tools + Reasoning Loop — it observes, thinks, acts, and repeats until the task is done.

**What you learn**:
- The agent formula: LLM + Tools + Prompt
- ReAct pattern (Reason + Act)
- Tool binding with LangChain
- When to use an agent vs a simple chain

**Prerequisites**: 01, 02, 03 (need LLM + function calling + prompts)

**Repo folder**: `2_What_Makes_An_Agent/`

---

### 06 — Multi-Agent Frameworks (`06_multi_agent_frameworks.md`)

**One-line summary**: Multiple specialized agents working together can solve problems that a single agent can't — LangChain and CrewAI are two ways to build these teams.

**What you learn**:
- LangChain agent executor
- CrewAI: Agent, Task, Crew concepts
- Sequential vs hierarchical agent orchestration
- Custom orchestration with pure Python (no framework)
- Tool integration with DuckDuckGo, Tavily, etc.

**Prerequisites**: 01, 02, 03, 05 (need to understand single agents first)

**Repo folder**: `3_Multi_AgenticAI_Frameworks/`

---

### 07 — Self-Reflection & Critique (`07_self_reflection_and_critique.md`)

**One-line summary**: Self-reflection lets an agent critique its own output and improve it — turning a one-shot answer into a refined, high-quality result.

**What you learn**:
- Generate → Critique → Refine loop
- Rubric-based evaluation (scoring against criteria)
- Reflection memory (learning from past mistakes)
- Constitutional AI (principle-based self-correction)

**Prerequisites**: 01, 03, 05 (need LLM + prompts + agent basics)

**Repo folder**: `5_Self_Reflection/`

---

### 08 — Sentiment Analysis (`08_sentiment_analysis.md`)

**One-line summary**: Sentiment analysis determines whether text is positive, negative, or neutral — and LLMs can do it better than traditional ML because they understand context and sarcasm.

**What you learn**:
- Three approaches: rule-based, ML classification, LLM-based
- Binary, ternary, fine-grained, and aspect-based sentiment
- Using Pydantic + LLM for structured sentiment extraction
- How sentiment analysis powers the Review Agent in the capstone

**Prerequisites**: 01, 03, 04 (need LLM + prompts + structured output)

**Repo folder**: `mini_project_sprint/product_recommendation_agent.ipynb` (Part 5.5)

---

### 10 — RAG Fundamentals (`10_rag_fundamentals.md`)

**One-line summary**: RAG grounds LLM answers in external documents — load, chunk, embed, store, retrieve, then generate.

**What you learn**:
- The 6-step RAG pipeline
- Document loaders: `TextLoader`, `PyPDFLoader`, `WebBaseLoader`, `WikipediaLoader`
- Text splitting with `RecursiveCharacterTextSplitter`
- Embedding models: HuggingFace MiniLM and OpenAI embeddings
- Retrieving chunks and generating grounded answers

**Prerequisites**: 01, 03 (need LLM + prompt basics)

**Repo folder**: `7_RAG/`

---

### 11 — Vector Databases & FAISS (`11_vector_databases_and_faiss.md`)

**One-line summary**: Vector databases store text as embeddings so you can search by semantic meaning instead of exact keywords.

**What you learn**:
- What embeddings are and why they work
- Cosine similarity vs Euclidean (L2) distance
- Building FAISS indexes (native and via LangChain)
- Adding, searching, saving, and loading vector stores
- When to use FAISS vs managed vector databases

**Prerequisites**: 10 (RAG fundamentals)

**Repo folder**: `7_RAG/3_FAISS/`

---

### 12 — Concept Flashcards & Relationship Map (`12_flashcards.md`)

**One-line summary**: 40 connected flashcards that explain not just *what* each concept is, but *how it relates* to the others.

**What you learn**:
- Core definitions (LLM, token, embedding, agent, RAG)
- Prompt engineering and structured output patterns
- How tools, function calling, and agents connect
- Why RAG needs embeddings and vector databases
- Cross-concept questions that test real understanding

**Prerequisites**: ALL previous notes (01-11) — best used as a review tool

**Repo folder**: `study_notes/`

---

### 09 — Building Product Recommendation Agent (`09_building_product_recommendation_agent.md`)

**One-line summary**: The capstone project — a 4-agent pipeline (Discovery → Specs → Reviews → Recommendation) that recommends products using everything you've learned.

**What you learn**:
- The 3-piece agent pattern: Prompt + Chain + Invoke
- 4-agent pipeline with data flow between agents
- Tavily and DuckDuckGo search integration
- Weighted scoring system (Performance 30%, Value 25%, Reviews 20%, AI Readiness 25%)
- Guardrails (allowed categories, budget validation, data source restrictions)
- Custom orchestration without any framework

**Prerequisites**: ALL previous notes (01-11)

**Repo folder**: `mini_project_sprint/`

---

## How to Use These Notes

### Step 1: Read in Order (Notes 01-11)
Each note builds on the previous. Don't skip ahead — concepts compound.

### Step 2: Try Every Code Example
Don't just read the code — run it. Modify it. Break it. Fix it.

### Step 3: Do the Practice Exercises
Each note has 3-5 exercises. These are where real learning happens.

### Step 4: Build the Capstone (Note 09)
After completing 01-11, work through the product recommendation agent. This is where everything clicks.

### Step 5: Extend It
Once the capstone works, try:
- Adding a new agent to the pipeline
- Changing the product category
- Swapping Tavily for DuckDuckGo (or vice versa)
- Building your own multi-agent system from scratch
- Adding a RAG layer to the product recommendation agent (e.g., retrieve from product manuals)

---

## Quick Reference: Concept → Note → Repo Folder

```
CONCEPT                          NOTE                                    REPO FOLDER
─────────────────────────────────────────────────────────────────────────────────────
LLMs, tokens, temperature       01_genai_and_llm_fundamentals.md                    intro.ipynb, 1_Introduction/
Function calling, tool schemas  02_function_calling_and_tools.md                  2_What_Makes_An_Agent/
Zero-shot, few-shot, CoT        03_prompt_engineering.md                4_Prompt_Essentials/
Pydantic, BaseModel, Field      04_pydantic_and_structured_output.md        mini_project_sprint/
Agent = LLM + Tools + Prompt    05_what_makes_an_agent.md               2_What_Makes_An_Agent/
LangChain agents, CrewAI        06_multi_agent_frameworks.md            3_Multi_AgenticAI_Frameworks/
Generate-critique-refine        07_self_reflection_and_critique.md                   5_Self_Reflection/
Sentiment analysis (LLM-based)  08_sentiment_analysis.md                mini_project_sprint/
RAG pipeline (load-chunk-embed) 10_rag_fundamentals.md                   7_RAG/
Vector DBs & similarity search  11_vector_databases_and_faiss.md        7_RAG/3_FAISS/
Concept flashcards & relations  12_flashcards.md                        study_notes/
4-agent product recommender     09_building_product_recommendation_agent.md  mini_project_sprint/
```

---

## Dependency Map

Which notes depend on which? Use this if you want to jump to a specific topic:

```
01 GenAI & LLMs ──────────────────────────────────────────────┐
 │                                                             │
 ├──▶ 02 Function Calling                                      │
 │     │                                                       │
 ├──▶ 03 Prompt Engineering                                    │
 │     │                                                       │
 │     ├──▶ 04 Pydantic & Structured Output                    │
 │     │                                                       │
 ├──▶ 05 What Makes an Agent (needs 01, 02, 03)               │
 │     │                                                       │
 │     ├──▶ 06 Multi-Agent Frameworks (needs 05)               │
 │     │                                                       │
 │     ├──▶ 07 Self-Reflection (needs 05)                      │
 │     │                                                       │
 │     └──▶ 08 Sentiment Analysis (needs 03, 04)               │
 │                                                             │
 ├──▶ 10 RAG Fundamentals (needs 01, 03)                       │
 │     │                                                       │
 │     └──▶ 11 Vector Databases & FAISS (needs 10)             │
 │                                                             │
 ├──▶ 12 Concept Flashcards (review of 01-11)                  │
 │                                                             │
 └──▶ 09 Capstone Project (needs ALL of 01-11) ◀──────────────┘
```

---

## Key Patterns to Remember

These patterns appear over and over throughout the course:

**Pattern 1: LLM Call**
```python
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
response = llm.invoke("Your question here")
```

**Pattern 2: Prompt + Chain**
```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert at {role}"),
    ("human", "{user_input}")
])
chain = prompt | llm
result = chain.invoke({"role": "...", "user_input": "..."})
```

**Pattern 3: Structured Output**
```python
chain = prompt | llm.with_structured_output(MyPydanticSchema)
result = chain.invoke({"key": "value"})  # returns MyPydanticSchema object
```

**Pattern 4: Agent Pipeline**
```python
result_1 = agent_1_chain.invoke(user_input)
result_2 = agent_2_chain.invoke(format(result_1))
result_3 = agent_3_chain.invoke(format(result_1, result_2))
```

**Pattern 5: RAG Pipeline**
```python
chunks = splitter.split_documents(docs)
vector_store = FAISS.from_documents(chunks, embeddings)
retrieved = vector_store.similarity_search(question, k=3)
answer = llm.invoke(prompt.format(context=retrieved, question=question))
```

---

## Tech Stack Used in This Course

- **Python 3.13** — The programming language
- **LangChain** — Framework for building LLM applications
- **OpenAI GPT (gpt-4o-mini / gpt-5)** — The LLM brain
- **Groq** — Alternative fast LLM provider
- **Pydantic** — Data validation and structured output
- **Tavily** — AI-optimized web search API
- **DuckDuckGo** — Free web search (no API key)
- **CrewAI** — Multi-agent orchestration framework
- **FAISS** — In-memory vector similarity search
- **sentence-transformers / HuggingFace** — Local embedding models
- **python-dotenv** — Environment variable management

---

**Start here** → `01_genai_and_llm_fundamentals.md`

**End here** → `09_building_product_recommendation_agent.md`

Good luck on your Agentic AI journey!
