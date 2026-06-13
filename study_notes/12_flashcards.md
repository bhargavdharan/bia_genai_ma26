# 12 — Concept Flashcards & Relationship Map

> **Purpose:** Quick-review cards that connect ideas instead of isolating them. Each card tells you **what** a concept is, **why** it exists, and **what it relates to**.

---

## How to Use These Flashcards

1. **Read the question.** Cover the answer.
2. **Answer in your own words.** Then check.
3. **Read the "Relativity" section.** This is the most important part — it tells you where the concept lives in the bigger picture.
4. **If you get it wrong, trace back.** Use the "Prerequisite concepts" link.

---

## Big-Picture Concept Map

```
                    [User Question]
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
   [Prompt Engineering] [Tools]      [Retrieval / RAG]
          │               │               │
          ▼               ▼               ▼
        [LLM] ◄─────► [Function Calling] [Embeddings]
          │                                    │
          ▼                                    ▼
   [Structured Output]                  [Vector DB / FAISS]
          │                                    │
          ▼                                    ▼
      [Agent] ◄──────────────────────── [Grounded Knowledge]
          │
          ▼
   [Multi-Agent System]
```

**Takeaway:** Almost everything flows into or out of the **LLM**. The three big superpowers you add are:
1. **Better control** → prompt engineering
2. **External capabilities** → tools / function calling
3. **External knowledge** → RAG / embeddings / vector DBs

---

## Foundation Concepts

### Card 1 — Generative AI (GenAI)

**Q:** What is Generative AI?

**A:** AI that creates new content — text, images, audio, code — by learning patterns from training data and generating outputs that look like they could belong to that data.

**Relativity:**
- **Parent of:** LLMs, image generators, code assistants
- **Depends on:** machine learning, deep learning, neural networks
- **Leads to:** prompts, agents, RAG

**Why it matters:** It shifts AI from "classify this" to "create something new."

---

### Card 2 — Large Language Model (LLM)

**Q:** What is an LLM?

**A:** A neural network trained on huge amounts of text to predict the next token. Examples: GPT-4o, Llama 3, Mistral.

**Relativity:**
- **A type of:** Generative AI
- **Powered by:** transformers, attention mechanism
- **Controlled by:** prompts, temperature, top-p
- **Extended by:** tools, RAG, agents

**Why it matters:** It is the "brain" at the center of most GenAI applications.

---

### Card 3 — Token

**Q:** What is a token?

**A:** A small piece of text — can be a word, part of a word, or punctuation. LLMs read and generate one token at a time.

**Relativity:**
- **Building block of:** LLM input and output
- **Affects:** cost, context window, speed
- **Related to:** inference, embeddings

**Why it matters:** Tokens are the unit of currency for LLMs. "1000 tokens" ≈ 750 words.

---

### Card 4 — Inference

**Q:** What is inference?

**A:** Running a trained model to get predictions. For LLMs, it means feeding a prompt and generating a response token by token.

**Relativity:**
- **Opposite of:** training
- **Uses:** model weights, prompt, sampling parameters
- **Produces:** tokens / text / structured output

**Why it matters:** Training is expensive and done once; inference is what users actually experience.

---

### Card 5 — Temperature

**Q:** What does temperature control?

**A:** How creative vs deterministic the LLM output is. Low temperature → safe, predictable. High temperature → random, creative.

**Relativity:**
- **A sampling parameter** alongside top-p, top-k
- **Works during:** inference
- **Balances:** creativity vs consistency

**Rule of thumb:**
- `0.0` → code, math, structured output
- `0.3–0.5` → balanced answers
- `0.8+` → brainstorming, creative writing

---

## Prompt Engineering

### Card 6 — Prompt

**Q:** What is a prompt?

**A:** The input text you give to an LLM. It can include instructions, context, examples, and the actual question.

**Relativity:**
- **Goes into:** LLM
- **Made better by:** prompt templates, few-shot examples, roles
- **Affects:** output quality more than model choice sometimes

**Why it matters:** A good prompt is the cheapest way to improve LLM output.

---

### Card 7 — Prompt Template

**Q:** What is a prompt template?

**A:** A reusable prompt with placeholders like `{topic}` or `{audience}`. At runtime you fill in the values.

**Relativity:**
- **Built on:** prompt engineering
- **Used in:** chains, agents, RAG
- **Example:** `"Explain {topic} to a {audience}"`

**Why it matters:** It separates prompt structure from runtime data.

---

### Card 8 — Role Prompting

**Q:** What is role prompting?

**A:** Telling the model WHO it is, e.g., "You are a patient science teacher."

**Relativity:**
- **Type of:** prompt engineering
- **Changes:** tone, depth, examples, format
- **Often paired with:** system messages

**Why it matters:** The same answer feels different when framed by a role.

---

### Card 9 — Few-Shot Prompting

**Q:** What is few-shot prompting?

**A:** Giving the model a few examples of input/output pairs before asking it to do the task.

**Relativity:**
- **Type of:** prompt engineering
- **Helps with:** classification, formatting, style matching
- **Alternative to:** fine-tuning

**Why it matters:** Examples teach the model the pattern faster than instructions alone.

---

## Structured Output

### Card 10 — Pydantic

**Q:** What is Pydantic?

**A:** A Python library for data validation using typed classes. You define fields with types and descriptions.

**Relativity:**
- **Used for:** structured output from LLMs
- **Replaces:** manual regex parsing of LLM text
- **Works with:** `llm.with_structured_output(Model)`

**Why it matters:** It turns messy text into reliable, typed Python objects.

---

### Card 11 — Structured Output

**Q:** What is structured output?

**A:** Forcing the LLM to return data in a predictable format like JSON or a Pydantic object.

**Relativity:**
- **Solves:** parsing instability
- **Enables:** chaining, database storage, API responses
- **Depends on:** Pydantic, JSON Schema

**Why it matters:** You cannot build reliable software on free-form text alone.

---

## Tools & Function Calling

### Card 12 — Function Calling

**Q:** What is function calling?

**A:** The LLM outputs a JSON object describing which tool to call and with what arguments. Your code executes the tool.

**Relativity:**
- **Gives the LLM:** hands
- **Uses:** tool descriptions (name, args, purpose)
- **Leads to:** agents

**Critical point:** The LLM does **not** run the tool. Your code does.

---

### Card 13 — Tool

**Q:** What is a tool in the context of agents?

**A:** A Python function wrapped with metadata (name + description) that an LLM can decide to invoke.

**Relativity:**
- **Called via:** function calling
- **Examples:** calculator, search, database query, API call
- **Enables:** agents

**Why it matters:** Tools extend the LLM beyond its training data.

---

### Card 14 — Agent

**Q:** What is an agent?

**A:** An LLM placed in a loop where it can decide to use tools, observe results, and reason until it produces a final answer.

**Relativity:**
- **Built from:** LLM + tools + prompt + loop
- **Uses:** function calling
- **Pattern:** ReAct (Reason + Act)
- **Can become:** multi-agent system

**Why it matters:** Agents turn a single question into a multi-step problem-solving process.

---

### Card 15 — ReAct

**Q:** What is ReAct?

**A:** A prompting pattern where the model alternates between **Reasoning** ("I need to calculate this") and **Acting** (calling a tool).

**Relativity:**
- **Agent pattern**
- **Uses:** function calling, tool descriptions
- **Loop:** thought → action → observation → thought → ...

**Why it matters:** It makes the agent's thinking visible and controllable.

---

## Embeddings & Vector Search

### Card 16 — Embedding

**Q:** What is an embedding?

**A:** A numerical vector that captures the meaning of text, images, or other data. Similar meanings → similar vectors.

**Relativity:**
- **Input:** text / image / audio
- **Output:** vector of numbers
- **Used in:** similarity search, RAG, clustering
- **Produced by:** embedding models like `all-MiniLM-L6-v2`

**Why it matters:** Embeddings let computers compare meaning mathematically.

---

### Card 17 — Cosine Similarity

**Q:** What is cosine similarity?

**A:** A measure of how similar two vectors are based on the angle between them. Score ranges from -1 to 1; higher means more similar.

**Relativity:**
- **Used with:** embeddings
- **Alternative to:** Euclidean distance, dot product
- **Common in:** semantic search, RAG

**Why it matters:** It focuses on direction (meaning), not magnitude (vector length).

---

### Card 18 — Vector Database

**Q:** What is a vector database?

**A:** A database optimized for storing vectors and searching them by similarity quickly.

**Relativity:**
- **Stores:** embeddings
- **Searches with:** cosine similarity, L2, dot product
- **Examples:** FAISS, Chroma, Pinecone, Qdrant, Weaviate
- **Used in:** RAG, recommendation systems

**Why it matters:** Comparing millions of vectors one-by-one is too slow; vector DBs make it fast.

---

### Card 19 — FAISS

**Q:** What is FAISS?

**A:** A library from Meta for efficient similarity search of dense vectors. Often used locally with LangChain.

**Relativity:**
- **Type of:** vector search library / vector DB
- **Uses:** embeddings
- **Works with:** LangChain `FAISS.from_documents()`
- **Good for:** local RAG prototypes

**Why it matters:** It is free, fast, and works offline.

---

## RAG

### Card 20 — RAG (Retrieval-Augmented Generation)

**Q:** What is RAG?

**A:** A technique where the LLM answers a question using retrieved documents as context, rather than relying only on training memory.

**Relativity:**
- **Combines:** retrieval + generation
- **Uses:** embeddings, vector DB, LLM
- **Pipeline:** load → chunk → embed → store → retrieve → generate
- **Solves:** hallucinations, outdated knowledge, private data

**Why it matters:** It grounds the LLM in real, current, or private documents.

---

### Card 21 — Chunking

**Q:** What is chunking?

**A:** Splitting long documents into smaller pieces before embedding them.

**Relativity:**
- **Step in:** RAG pipeline
- **Controls:** context size, retrieval precision
- **Common tool:** `RecursiveCharacterTextSplitter`
- **Trade-off:** small chunks = precise; large chunks = more context

**Why it matters:** Embedding an entire book as one vector loses detail. Chunking preserves it.

---

### Card 22 — Retrieval

**Q:** What is retrieval in RAG?

**A:** Finding the most relevant chunks from a vector store for a given query.

**Relativity:**
- **Step in:** RAG pipeline
- **Uses:** vector DB, embeddings, similarity metric
- **Feeds into:** LLM prompt as context

**Why it matters:** Bad retrieval = bad RAG, no matter how good the LLM is.

---

### Card 23 — Chunk Overlap

**Q:** What is chunk overlap?

**A:** The number of characters shared between consecutive chunks so that no sentence or idea is cut off at the boundary.

**Relativity:**
- **Chunking parameter**
- **Balances:** context continuity vs redundancy

**Why it matters:** Without overlap, a sentence split across chunks may be unretrievable.

---

## Multi-Agent & Orchestration

### Card 24 — Multi-Agent System

**Q:** What is a multi-agent system?

**A:** A system where multiple agents, each with a specialized role, collaborate to solve a complex problem.

**Relativity:**
- **Extension of:** single agent
- **Uses:** structured output to pass data between agents
- **Examples:** researcher → writer → critic → editor

**Why it matters:** Specialists are often better than one generalist.

---

### Card 25 — CrewAI / Multi-Agent Frameworks

**Q:** What do multi-agent frameworks like CrewAI provide?

**A:** They manage roles, tasks, tools, and communication flow between agents so you do not have to write the orchestration manually.

**Relativity:**
- **Built on:** agent + tool concepts
- **Simplify:** multi-agent coordination
- **Require:** clear task definitions

**Why it matters:** Frameworks reduce boilerplate and enforce patterns.

---

## Self-Reflection & Evaluation

### Card 26 — Self-Reflection

**Q:** What is self-reflection in agents?

**A:** The agent evaluates its own output and either improves it or stops when quality is good enough.

**Relativity:**
- **Agent improvement technique**
- **Uses:** a second LLM call or a critic agent
- **Related to:** evaluation, iterative refinement

**Why it matters:** It helps agents catch their own mistakes.

---

### Card 27 — Hallucination

**Q:** What is hallucination?

**A:** When an LLM generates plausible-sounding but false or unsupported information.

**Relativity:**
- **Problem solved by:** RAG, grounding, citations, self-reflection
- **More likely with:** high temperature, out-of-date knowledge, ambiguous prompts

**Why it matters:** Hallucinations destroy trust in AI systems.

---

## Architecture & Patterns

### Card 28 — Chain

**Q:** What is a chain in LangChain?

**A:** A sequence of components connected together, e.g., `prompt | llm | parser`.

**Relativity:**
- **Building block of:** LangChain apps
- **Can include:** prompts, LLMs, tools, retrievers
- **Leads to:** agents (chains in a loop)

**Why it matters:** Chains make pipelines readable and reusable.

---

### Card 29 — Corpus

**Q:** What is a corpus?

**A:** A collection of documents used for training, search, or retrieval.

**Relativity:**
- **Used in:** RAG, embeddings, information retrieval
- **Can be:** web pages, PDFs, notes, books

**Why it matters:** The quality of your corpus determines the quality of your RAG answers.

---

### Card 30 — Context Window

**Q:** What is the context window?

**A:** The maximum amount of text the LLM can process in one call, measured in tokens.

**Relativity:**
- **Limits:** prompt size + retrieved chunks
- **Reason for:** chunking
- **Affects:** cost and memory

**Why it matters:** You cannot feed an entire library into one prompt; you must retrieve selectively.

---

## Cross-Concept Questions (Harder)

### Card 31

**Q:** Why do agents need tools if LLMs already know a lot?

**A:** LLMs are frozen at training time. Tools give them access to current data, computation, APIs, and private documents.

**Relativity:** LLM knowledge ≠ real-time / external / actionable knowledge.

---

### Card 32

**Q:** Why does RAG use embeddings instead of keyword search?

**A:** Keyword search only finds exact words. Embeddings find similar meaning even when different words are used.

**Relativity:** embeddings → vector DB → semantic retrieval → grounded generation.

---

### Card 33

**Q:** What is the difference between function calling and a tool?

**A:** A tool is the actual Python function. Function calling is the LLM's ability to request that the tool be executed.

**Relativity:** tool = implementation; function calling = interface/contract.

---

### Card 34

**Q:** Why is chunk size important in RAG?

**A:** Too small → loses context. Too large → includes irrelevant text and wastes context window.

**Relativity:** chunking ↔ retrieval precision ↔ context window ↔ answer quality.

---

### Card 35

**Q:** How do Pydantic and function calling work together?

**A:** Pydantic defines the shape of structured output. Function calling also uses schemas to describe tool inputs. Both rely on JSON Schema.

**Relativity:** Pydantic output schema ↔ function input schema ↔ LLM schema understanding.

---

### Card 36

**Q:** When should you use an agent instead of a simple chain?

**A:** Use an agent when the number or order of steps is not fixed and the model must decide what to do next based on intermediate results.

**Relativity:** chain = fixed pipeline; agent = dynamic decision loop.

---

### Card 37

**Q:** Why would you build a multi-agent system instead of one agent with many tools?

**A:** Specialists with focused prompts and tools are easier to debug, maintain, and scale than one agent trying to do everything.

**Relativity:** single agent complexity → role separation → cleaner orchestration.

---

### Card 38

**Q:** What happens if retrieval in a RAG system returns bad chunks?

**A:** The LLM generates bad or hallucinated answers because it is grounded in the wrong context.

**Relativity:** retrieval quality is the ceiling for RAG answer quality.

---

### Card 39

**Q:** How does temperature affect structured output?

**A:** High temperature can make the model ignore format instructions. For structured output, use low temperature (0.0–0.3).

**Relativity:** temperature → randomness → format adherence → structured output reliability.

---

### Card 40

**Q:** Why is it important to keep API keys out of notebooks?

**A:** Leaked keys can be stolen and used to run up costs or access your accounts. GitHub also blocks pushes containing secrets.

**Relativity:** security → `.env` files → environment variables → safe collaboration.

---

## Suggested Study Routine

| Day | Activity |
|---|---|
| Day 1 | Foundation cards (1–5) |
| Day 2 | Prompt + structured output cards (6–11) |
| Day 3 | Tools + agents cards (12–15) |
| Day 4 | Embeddings + vector DB cards (16–19) |
| Day 5 | RAG cards (20–23) |
| Day 6 | Multi-agent + reflection cards (24–27) |
| Day 7 | Cross-concept cards (31–40) |

**Final check:** If you can explain each card's "Relativity" section without looking, you understand the subject as a connected system, not isolated facts.
