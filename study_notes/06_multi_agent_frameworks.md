# 06 — Multi-Agent AI Frameworks

## What is it?
A **multi-agent system** is a group of specialized AI agents that collaborate to solve complex problems — each agent is an expert at one thing, and they pass data between each other like teammates in a relay race.

---

## Why does it matter?
One agent trying to do everything is like one employee doing sales, engineering, marketing, and accounting. It works for small tasks, but it breaks down fast. In production AI systems (customer support pipelines, research assistants, product recommendation engines), you split responsibilities across multiple agents. Each agent focuses on what it's best at, and the overall system is more reliable, easier to debug, and produces better results.

---

## How does it work? (Under the Hood)

### Why Multi-Agent? The Division of Labor

```
SINGLE AGENT (one brain, many hats):
┌─────────────────────────────────────┐
│  "Find products, research specs,    │
│   read reviews, score them, and     │
│   make a recommendation"            │
│                                     │
│   → Too much for one LLM call      │
│   → Context window overload         │
│   → Inconsistent results            │
└─────────────────────────────────────┘

MULTI-AGENT (specialized team):
┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
│ Agent 0  │──▶│ Agent 1  │──▶│ Agent 2  │──▶│ Agent 3  │
│ SCOUT    │   │ ENGINEER │   │ CRITIC   │   │ ADVISOR  │
│ Find     │   │ Get      │   │ Analyze  │   │ Score &  │
│ products │   │ specs    │   │ reviews  │   │ recommend│
└──────────┘   └──────────┘   └──────────┘   └──────────┘
     ↓              ↓              ↓              ↓
  Pydantic       Pydantic       Pydantic       Pydantic
  output         output         output         output
```

### The Three Major Frameworks

Here's how the main frameworks compare:

#### 1. LangGraph — Stateful Workflows

**Philosophy**: "AI workflows are graphs. Nodes are steps, edges are transitions."

```
Best for:
- Complex workflows with branching logic
- Retry loops and error recovery
- Human-in-the-loop (HITL) approval steps
- Production systems that need reliability

How it works:
┌───────┐     ┌─────────┐     ┌──────────┐
│ Start │────▶│ Search  │────▶│ Reflect  │
│ Node  │     │ Node    │     │ Node     │──┐
└───────┘     └─────────┘     └──────────┘  │
                   ▲                         │
                   │      (retry if bad)     │
                   └─────────────────────────┘
```

Key concepts:
- **StateGraph** — your workflow is a graph with named nodes
- **Nodes** — each node is a function or agent that processes state
- **Edges** — connections between nodes (can be conditional)
- **State** — a shared dict that flows through the graph
- **Checkpoints** — save/resume state (great for HITL)

```python
# Pseudo-code from the repo (practical.ipynb)
# from langgraph.graph import StateGraph
# graph = StateGraph()
# graph.add_node("start", StartNode())
# graph.add_node("search", SearchNode())
# graph.add_node("reflect", ReflectNode())
# graph.set_entry_point("start")
# graph.add_edge("start", "search")
# graph.add_edge("search", "reflect")
# graph.add_edge("reflect", "search")  # retry loop
```

#### 2. CrewAI — Role-Based Collaboration

**Philosophy**: "Think of agents as team members with roles, goals, and backstories."

```
Best for:
- Rapid prototyping of multi-agent systems
- When agents need clear role definitions
- Sequential or parallel task execution
- Less boilerplate than LangGraph

How it works:
┌─────────────────────────────────────────────┐
│                   CREW                       │
│                                              │
│  ┌──────────────┐    ┌──────────────┐       │
│  │ Agent:       │    │ Agent:       │       │
│  │ "Researcher" │    │ "Writer"     │       │
│  │ Role: ...    │    │ Role: ...    │       │
│  │ Goal: ...    │    │ Goal: ...    │       │
│  │ Backstory: . │    │ Backstory: . │       │
│  └──────┬───────┘    └──────┬───────┘       │
│         │                    │                │
│  ┌──────▼───────┐    ┌──────▼───────┐       │
│  │ Task 1:      │───▶│ Task 2:      │       │
│  │ "Research X" │    │ "Write about │       │
│  │              │    │  X findings" │       │
│  └──────────────┘    └──────────────┘       │
│                                              │
│  Process: sequential | parallel              │
└─────────────────────────────────────────────┘
```

Key concepts:
- **Agent** — has a `role`, `goal`, `backstory`, and optional tools
- **Task** — a specific job assigned to an agent
- **Crew** — a collection of agents + tasks + process type
- **Process** — `sequential` (one after another) or `hierarchical`
- **Delegation** — agents can delegate sub-tasks to each other

#### 3. AutoGen — Multi-Agent Conversations

**Philosophy**: "Agents talk to each other in a group chat until the task is done."

```
Best for:
- Code generation and review workflows
- Iterative refinement (generate → critique → improve)
- When agents need to debate or build on each other's work
- Research and brainstorming tasks

How it works:
┌─────────────────────────────────────────────┐
│            RoundRobinGroupChat               │
│                                              │
│  ┌──────────┐        ┌──────────┐           │
│  │ Primary  │───────▶│ Critic   │           │
│  │ Agent    │◀───────│ Agent    │           │
│  │          │        │          │           │
│  │ "Write   │        │ "Review  │           │
│  │  a poem" │        │  & give  │           │
│  │          │        │  feedback"│           │
│  └──────────┘        └──────────┘           │
│                                              │
│  Termination: when critic says "APPROVE"    │
└─────────────────────────────────────────────┘
```

Key concepts:
- **AssistantAgent** — an LLM-powered agent with a system message
- **UserProxyAgent** — represents the human (can auto-execute code)
- **GroupChat** — multiple agents take turns in a conversation
- **Termination conditions** — stop when a keyword appears or after N rounds

### When to Use Which Framework

```
Decision Guide:

Need stateful workflows with retries?
  → LangGraph

Need role-based agents with minimal code?
  → CrewAI

Need agents that debate/iterate on content?
  → AutoGen

Need full control, no framework overhead?
  → Custom orchestration (pure Python)

Building a production pipeline?
  → LangGraph or Custom

Building a quick prototype?
  → CrewAI
```

### Custom Orchestration — No Framework

Sometimes frameworks add complexity you don't need. The mini project sprint in this repo uses **pure Python orchestration**:

```
Custom Pipeline (what the repo does):

User Query
    │
    ▼
┌──────────────┐     DiscoveryOutput
│  Agent 0:    │────────────────────────▶┐
│  Discovery   │  (Pydantic object)      │
└──────────────┘                         │
                                         ▼
                                  ┌──────────────┐     SpecificationOutput
                                  │  Agent 1:    │────────────────────────▶┐
                                  │  Specs       │  (Pydantic object)      │
                                  └──────────────┘                         │
                                                                           ▼
                                                                    ┌──────────────┐
                                                                    │  Agent 2:    │
                                                                    │  Reviews     │──▶ Agent 3
                                                                    └──────────────┘    Recommendation
```

The key insight: data flows between agents via **Pydantic objects**. Each agent returns a structured output that the next agent consumes. No framework needed — just functions calling functions.

### The Framework Trade-Off

```
MORE FEATURES ←────────────────────────────→ MORE CONTROL
   LangGraph          CrewAI         AutoGen         Pure Python
   
   ✅ State mgmt     ✅ Easy roles   ✅ Conversations  ✅ Full control
   ✅ Retries         ✅ Delegation   ✅ Code exec      ✅ No dependencies
   ✅ HITL            ✅ Quick setup  ✅ Iteration      ✅ Lightweight
   
   ❌ Complex API    ❌ Less control ❌ Chatty          ❌ Build everything
   ❌ Steep learning ❌ Opinionated  ❌ Hard to debug   ❌ No built-in retry
```

---

## How does it connect to other topics?

- **See: `05_what_makes_an_agent.md`** — Each agent in a multi-agent system is built using the same agent architecture (LLM + Tools + Memory + Loop). Multi-agent is just composing multiple agents together.
- **See: `09_building_product_recommendation_agent.md`** — The mini project sprint is a working example of custom multi-agent orchestration with 4 agents.
- **See: `07_self_reflection_and_critique.md`** — Self-reflection is a pattern used WITHIN agents (generate → critique → refine). AutoGen's primary/critic pattern is a multi-agent version of this.
- **See: `02_function_calling_and_tools.md`** — Tools are what give each agent its specialization. Without function calling, agents would all just be text generators.

---

## Code Examples

### Example 1: LangChain agent with tools (from the repo)

This is the core agent from `3_Multi_AgenticAI_Frameworks/practical.ipynb`:

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
        "crewai": "CrewAI focuses on role-based multi-agent collaboration.",
        "autogen": "AutoGen enables multi-agent conversations."
    }
    query = query.lower()
    for key, value in corpus.items():
        if key in query:
            return value
    return "No relevant results found."

tools = [
    Tool(name="calculator", func=safe_calculate,
         description="Use this to perform mathematical calculations."),
    Tool(name="search", func=search_corpus,
         description="Searches a local knowledge corpus for information.")
]

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
agent = create_agent(model=model, tools=tools)

# The agent decides which tool to use based on the query
response = agent.invoke({
    "messages": [{"role": "user", "content": "What is langgraph"}]
})
print(response["messages"][-1].content)
# → "Langgraph is a tool designed for managing stateful workflows,
#    handling retries, and facilitating human-in-the-loop (HITL) processes."
```

Notice how the agent:
1. Received "What is langgraph"
2. Decided to use the `search` tool (not the calculator)
3. Passed "langgraph" to the search function
4. Got back the corpus entry
5. Formatted a nice human-readable answer

### Example 2: AutoGen two-agent conversation (from the repo)

This is from `3_Multi_AgenticAI_Frameworks/practical.ipynb` — a primary agent writes content, a critic reviews it:

```python
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Create the LLM client
model_client = OpenAIChatCompletionClient(model="gpt-4o-2024-08-06")

# Create the primary agent (the writer)
primary_agent = AssistantAgent(
    "primary",
    model_client=model_client,
    system_message="You are a helpful AI assistant.",
)

# Create the critic agent (the reviewer)
critic_agent = AssistantAgent(
    "critic",
    model_client=model_client,
    system_message="Provide constructive feedback. Respond with 'APPROVE' when satisfied.",
)

# Stop when the critic says "APPROVE"
text_termination = TextMentionTermination("APPROVE")

# Create a round-robin team (they take turns)
team = RoundRobinGroupChat(
    [primary_agent, critic_agent],
    termination_condition=text_termination
)

# Run the team
result = await team.run(task="Write a short poem about the fall season.")
```

What happens during execution:
```
Round 1: primary writes a poem
Round 2: critic gives feedback ("add more emotion, use metaphors")
Round 3: primary revises the poem based on feedback
Round 4: critic says "APPROVE" → conversation ends
```

This is the **generate → critique → refine** loop (See: `07_self_reflection_and_critique.md`), but with two separate agents instead of one.

### Example 3: Custom multi-agent pipeline (from mini_project_sprint)

The product recommendation system uses pure Python — no framework:

```python
from pydantic import BaseModel, Field
from typing import List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from tavily import TavilyClient

# ---- Pydantic schemas define what each agent returns ----
class DiscoveredProduct(BaseModel):
    brand: str = Field(description="Product manufacturer")
    model: str = Field(description="Product model name")
    specs_hint: str = Field(description="Brief specs from search")
    source: str = Field(description="URL where found")
    reason: str = Field(description="Why this matches the query")

class DiscoveryOutput(BaseModel):
    category: str = Field(description="Detected category")
    budget: str = Field(description="Detected budget range")
    use_case: str = Field(description="Detected use case")
    products: List[DiscoveredProduct] = Field(description="Up to 5 products")

class TechnicalSpecs(BaseModel):
    product_name: str = Field(description="Full product name")
    cpu: str = Field(description="Processor model")
    gpu: str = Field(description="Graphics card")
    ram: str = Field(description="RAM capacity and type")
    storage: str = Field(description="Storage capacity")
    display: str = Field(description="Display specs")
    price: str = Field(description="Approximate price")
    summary: str = Field(description="One-line summary")

class SpecificationOutput(BaseModel):
    specs: List[TechnicalSpecs] = Field(description="Specs for all products")

# ---- The pipeline: Agent 0 → Agent 1 → Agent 2 → Agent 3 ----
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# Agent 0: Discovery — search the web for products
def run_discovery_agent(user_query: str) -> DiscoveryOutput:
    search_results = tavily_client.search(query=user_query, max_results=5)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a product discovery agent..."),
        ("user", "Query: {query}\nSearch results: {results}")
    ])
    chain = prompt | llm.with_structured_output(DiscoveryOutput)
    return chain.invoke({"query": user_query, "results": search_results})

# Agent 1: Specs — takes Discovery output, researches specs
def run_spec_agent(discovery: DiscoveryOutput) -> SpecificationOutput:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a technical specification agent..."),
        ("user", "Products: {products}")
    ])
    chain = prompt | llm.with_structured_output(SpecificationOutput)
    return chain.invoke({"products": discovery.products})

# Run the pipeline
discovery = run_discovery_agent("Best laptop for AI under 1.5 lakh")
specs = run_spec_agent(discovery)  # Pydantic object flows agent-to-agent
# ... Agent 2 and 3 follow the same pattern
```

**Key design decisions:**
- **No framework** — just Python functions calling each other
- **Pydantic objects** — guarantee structured data between agents
- **`with_structured_output()`** — forces the LLM to return a Pydantic object
- **Each agent is independent** — you can test/debug each one separately

### Example 4: LangGraph pseudo-code (from the repo)

This sketch shows how LangGraph creates a retry loop:

```python
# from langgraph.graph import StateGraph

# graph = StateGraph()
# graph.add_node("start", StartNode())
# graph.add_node("search", SearchNode())
# graph.add_node("reflect", ReflectNode())
# graph.set_entry_point("start")
# graph.add_edge("start", "search")
# graph.add_edge("search", "reflect")
# graph.add_edge("reflect", "search")  # retry loop — keeps going until quality is good
```

The reflect node checks quality. If the result isn't good enough, it loops back to search. This is what makes LangGraph powerful for production systems — built-in retry logic.

---

## Common Mistakes

1. **Using a framework when you don't need one** — If your pipeline is linear (A → B → C → D), pure Python is simpler and easier to debug. Don't add LangGraph just because it sounds cool.

2. **Not defining clear agent boundaries** — Each agent should have ONE job. If an agent is doing 5 things, split it into 5 agents. The product recommendation project shows this perfectly: discover, spec, review, recommend.

3. **Ignoring data contracts between agents** — Without Pydantic schemas, Agent B has no idea what format Agent A's output will be in. Always define the structure of data flowing between agents.

4. **Making agents too chatty (AutoGen trap)** — In AutoGen, agents can go back and forth forever. Always set a termination condition (`TextMentionTermination`, max rounds, etc.) or your agents will chat into infinity.

5. **Not testing agents individually** — Before connecting agents into a pipeline, test each one in isolation. "Agent 2 is broken" is easier to debug than "the pipeline doesn't work."

6. **Mixing framework conventions** — Pick one framework and stick with it. Don't use LangGraph for Agent 1 and CrewAI for Agent 2. They have different state management approaches.

---

## Practice Exercises

1. **Build a custom 2-agent pipeline**: Create Agent A (researcher) that takes a topic and returns 3 key facts (as a Pydantic model), and Agent B (writer) that takes those facts and writes a paragraph. Use pure Python, no framework.

2. **AutoGen debate**: Set up a RoundRobinGroupChat with three agents — an optimist, a pessimist, and a moderator. Give them a topic like "Should AI replace human teachers?" The moderator should say "CONCLUDE" when both sides have been heard.

3. **Framework comparison**: Take the calculator + search agent from Example 1 and rewrite it using CrewAI (define a "Calculator Agent" and a "Research Agent" as a crew). Compare the code complexity.

4. **Add error recovery**: Take the custom pipeline from Example 3 and add error handling — if Agent 1 fails (returns empty specs), retry up to 3 times before falling back to a default response.

5. **Design a 3-agent system**: On paper, design a multi-agent system for "automated code review." Define the agents (e.g., Style Checker, Bug Finder, Security Auditor), their inputs/outputs (Pydantic schemas), and the flow between them. Then implement Agent 1 as a working prototype.
