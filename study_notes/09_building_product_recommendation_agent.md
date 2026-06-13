# 09 ΓÇö Building a Product Recommendation Agent (Capstone Project)

## What is it?

A **4-agent AI system** that takes a natural language query like "Best laptops for AI development under 150000 INR" and automatically discovers products on the web, researches their specs, analyzes user reviews, scores everything, and delivers a ranked recommendation ΓÇö all without any orchestration framework, just pure Python.

---

## Why does it matter?

This is the capstone project ΓÇö it brings together **every single concept** from the course into one working system. In the real world, companies build exactly this kind of multi-agent pipeline for product comparison, market research, customer support, and more. Understanding how to wire multiple specialized agents into a pipeline is the core skill of an Agentic AI engineer.

---

## How does it work? (Under the Hood)

### The Big Picture ΓÇö 4 Agents in a Pipeline

```
User Query: "Best laptops for AI development under 150000 INR"
      Γöé
      Γû╝
ΓöîΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÉ
Γöé  AGENT 0: Discovery Agent   Γöé  ΓåÉ "The Scout"
Γöé  - Searches the web         Γöé
Γöé  - LLM extracts products    Γöé
Γöé  - Output: DiscoveryOutput  Γöé
ΓööΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓö¼ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÿ
               Γöé  (product list)
               Γû╝
ΓöîΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÉ
Γöé  AGENT 1: Specification     Γöé  ΓåÉ "The Engineer"
Γöé  - Searches specs per item  Γöé
Γöé  - LLM extracts CPU/GPU/   Γöé
Γöé    RAM/Storage/Price        Γöé
Γöé  - Output: SpecificationOutput
ΓööΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓö¼ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÿ
               Γöé  (specs for each product)
               Γû╝
ΓöîΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÉ
Γöé  AGENT 2: Review Analysis   Γöé  ΓåÉ "The Critic"
Γöé  - Searches reviews per itemΓöé
Γöé  - LLM does sentiment      Γöé
Γöé    analysis (Pos/Mix/Neg)   Γöé
Γöé  - Extracts pros, cons,    Γöé
Γöé    issues, expert opinions  Γöé
Γöé  - Output: ReviewOutput     Γöé
ΓööΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓö¼ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÿ
               Γöé  (review analysis)
               Γû╝
ΓöîΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÉ
Γöé  AGENT 3: Recommendation    Γöé  ΓåÉ "The Advisor"
Γöé  - Receives ALL data from   Γöé
Γöé    Agent 0, 1, and 2        Γöé
Γöé  - Scores each product      Γöé
Γöé  - Ranks them               Γöé
Γöé  - Output: RecommendationOutput
ΓööΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÿ
               Γöé
               Γû╝
         Final Report:
         ≡ƒÅå Top Pick + Scores + Trade-offs + Verdict
```

### The 3-Piece Agent Pattern

Every agent in this project is built from exactly **3 pieces**. This is the fundamental pattern:

```python
# Piece 1: THE PROMPT ΓÇö Instructions for the LLM
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a product discovery expert. Extract up to 5 real products..."),
    ("human", "User Query:\n{user_query}\n\nSearch Results:\n{search_context}")
])

# Piece 2: THE CHAIN ΓÇö Connect prompt to LLM with structured output
chain = prompt | llm.with_structured_output(DiscoveryOutput)

# Piece 3: INVOKE ΓÇö Run it with actual data
result = chain.invoke({"user_query": "Best laptops...", "search_context": search_data})
```

That's it. Every agent follows this exact pattern. The only things that change are:
- The **prompt** (different instructions for each agent's job)
- The **schema** (different Pydantic model for each agent's output)
- The **input variables** (different data fed in)

---

### Agent 0: Discovery Agent (The Scout)

**Job**: Take user query ΓåÆ search the web ΓåÆ extract real product names

```python
# Schema: What Agent 0 returns
class DiscoveredProduct(BaseModel):
    brand: str = Field(description="Product manufacturer (e.g., ASUS, Lenovo)")
    model: str = Field(description="Product model name (e.g., ROG Strix G16)")
    specs_hint: str = Field(description="Brief specs from search (e.g., RTX 4060, 16GB)")
    source: str = Field(description="URL where this product was found")
    reason: str = Field(description="Why this product matches the user query")

class DiscoveryOutput(BaseModel):
    category: str = Field(description="Detected category (e.g., Laptops)")
    budget: str = Field(description="Detected budget (e.g., Under 1,50,000)")
    use_case: str = Field(description="Detected use case (e.g., AI Development)")
    products: List[DiscoveredProduct] = Field(description="List of products (up to 5)")

# Build and run
discovery_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a product discovery expert..."),
    ("human", "User Query:\n{user_query}\n\nSearch Results:\n{search_context}")
])
discovery_chain = discovery_prompt | llm.with_structured_output(DiscoveryOutput)

# Step 1: Search web first
search_context = search_products(f"{user_query} top models India {date.today().year}", 10)
# Step 2: LLM extracts products from raw search results
discovery_result = discovery_chain.invoke({
    "user_query": user_query,
    "search_context": search_context
})
```

**Key insight**: The LLM doesn't search. The `search_products()` function searches. The LLM just reads the messy search results and extracts clean, structured product data.

---

### Agent 1: Specification Agent (The Engineer)

**Job**: Take product list from Agent 0 ΓåÆ search specs for each ΓåÆ extract hardware details

```python
class TechnicalSpecs(BaseModel):
    product_name: str = Field(description="Full product name (Brand + Model)")
    cpu: str = Field(description="Processor model and speed")
    gpu: str = Field(description="Graphics card model and VRAM")
    ram: str = Field(description="RAM capacity and type (e.g., 16GB DDR5)")
    storage: str = Field(description="Storage capacity and type (e.g., 1TB SSD)")
    display: str = Field(description="Display size, resolution, panel type")
    battery: str = Field(description="Battery capacity and estimated life")
    weight: str = Field(description="Device weight in kg")
    price: str = Field(description="Approximate price in INR")
    summary: str = Field(description="One-line AI/ML focused summary")

class SpecificationOutput(BaseModel):
    specs: List[TechnicalSpecs] = Field(description="Specs for all products")
```

**Data flow**: `discovery_result.products` ΓåÆ loop through each ΓåÆ `search_specs(name)` ΓåÆ feed all to LLM ΓåÆ `SpecificationOutput`

```python
# Prepare product list string from Agent 0 output
product_list_str = "\n".join(f"- {p.brand} {p.model}" for p in discovery_result.products)

# Search specs for EACH product
all_research = ""
for p in discovery_result.products:
    name = f"{p.brand} {p.model}"
    all_research += f"\n--- {name} ---\n{search_specs(name)}\n"

# LLM extracts structured specs
spec_result = spec_chain.invoke({
    "product_list": product_list_str,
    "research_data": all_research
})
```

---

### Agent 2: Review Analysis Agent (The Critic)

**Job**: Search reviews ΓåÆ perform **sentiment analysis** ΓåÆ extract pros, cons, issues

```python
class ProductReview(BaseModel):
    product_name: str = Field(description="Full product name")
    overall_sentiment: str = Field(description="Positive, Mixed, or Negative")
    average_rating: str = Field(description="Estimated rating out of 5")
    pros: List[str] = Field(description="Top 3-5 praised features")
    cons: List[str] = Field(description="Top 3-5 common complaints")
    common_issues: List[str] = Field(description="Recurring problems")
    expert_opinion: str = Field(description="Expert reviewer summary")

class ReviewOutput(BaseModel):
    reviews: List[ProductReview] = Field(description="Review analysis for all products")
```

**How sentiment analysis works here** (LLM-based approach):

```
Traditional sentiment: "Is this positive or negative?"  ΓåÆ  "Positive"
Our approach:          "Analyze these reviews"          ΓåÆ  {
                                                            sentiment: "Positive",
                                                            rating: "4.3/5",
                                                            pros: ["Powerful GPU", "Good display"],
                                                            cons: ["Heavy", "Loud fans"],
                                                            issues: ["Overheating under load"],
                                                            expert_opinion: "Great for AI workloads"
                                                          }
```

This is **aspect-based sentiment analysis** ΓÇö not just positive/negative, but WHY.

See: **08_sentiment_analysis.md** for the full deep-dive on sentiment analysis approaches.

---

### Agent 3: Recommendation Agent (The Advisor)

**Job**: Take ALL data from previous agents ΓåÆ score ΓåÆ rank ΓåÆ give final verdict

```python
class ProductScore(BaseModel):
    product_name: str = Field(description="Full product name")
    performance_score: float = Field(description="Hardware capability (0-10)")
    value_score: float = Field(description="Value for money (0-10)")
    review_score: float = Field(description="User satisfaction (0-10)")
    ai_readiness_score: float = Field(description="AI/ML suitability (0-10)")
    overall_score: float = Field(description="Weighted overall (0-10)")
    rank: int = Field(description="Final rank (1 = best)")

class RecommendationOutput(BaseModel):
    top_pick: str = Field(description="#1 recommended product")
    top_pick_reason: str = Field(description="Why it's the top pick")
    scores: List[ProductScore] = Field(description="Scores for all products")
    trade_offs: str = Field(description="Trade-offs between top products")
    final_verdict: str = Field(description="Final advice (2-3 sentences)")
    confidence: str = Field(description="High, Medium, or Low")
```

**Key**: Agent 3 does NOT search the web. It only **reasons** over data already collected.

### The Scoring System

```
Overall Score = Performance(30%) + Value(25%) + Reviews(20%) + AI Readiness(25%)

Performance (30%):  CPU + GPU power
Value (25%):        How much you get per rupee
Reviews (20%):      What real users say (sentiment)
AI Readiness (25%): VRAM, RAM, compute for ML workloads
```

This weighting is defined in the prompt, and the LLM follows it when scoring.

---

### Data Flow Between Agents

This is the critical concept ΓÇö **output of one agent becomes input of the next**:

```
Agent 0 returns: discovery_result (DiscoveryOutput)
                     Γöé
                     Γö£ΓöÇΓöÇ discovery_result.products ΓåÆ Agent 1 gets product names to search
                     Γö£ΓöÇΓöÇ discovery_result.products ΓåÆ Agent 2 gets product names to search
                     Γöé
Agent 1 returns: spec_result (SpecificationOutput)
                     Γöé
Agent 2 returns: review_result (ReviewOutput)
                     Γöé
                     Γû╝
Agent 3 receives:  discovery_result + spec_result + review_result
                     Γöé
                     Γû╝
                 recommendation_result (RecommendationOutput)
```

In code, the handoff looks like this:

```python
# Agent 3 receives everything as formatted strings
disc_str = "\n".join(
    f"- {p.brand} {p.model}: {p.specs_hint} ({p.reason})"
    for p in discovery_result.products
)
spec_str = "\n".join(
    f"- {s.product_name}: CPU={s.cpu}, GPU={s.gpu}, RAM={s.ram}, Price={s.price}"
    for s in spec_result.specs
)
rev_str = "\n".join(
    f"- {r.product_name}: {r.overall_sentiment}, {r.average_rating}/5, Pros={r.pros[:3]}"
    for r in review_result.reviews
)

recommendation_result = recommendation_chain.invoke({
    "user_query": user_query,
    "discovery_data": disc_str,
    "spec_data": spec_str,
    "review_data": rev_str
})
```

---

### Search Tools: Tavily vs DuckDuckGo

The repo has two versions of this project:

**Tavily** (`product_recommendation_agent.ipynb`):
```python
from tavily import TavilyClient
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
results = tavily_client.search(query="...", max_results=5)
```
- Needs an API key (free tier available)
- Returns clean, AI-optimized results (title, URL, content snippet)
- Better quality for structured extraction

**DuckDuckGo** (`productrecomdagent.ipynb`):
```python
from langchain_community.tools import DuckDuckGoSearchResults
search = DuckDuckGoSearchResults(max_results=5)
results = search.invoke("...")
```
- Completely free, no API key needed
- Returns messier results
- Good for getting started quickly

---

### Guardrails

The system has built-in safety boundaries:

**Allowed Categories**: Electronics, Laptops, Mobile Phones, Monitors, Headphones, Home Appliances, Books

**Restricted Categories**: Weapons, Illegal Products, Restricted Substances, Counterfeit Products

**Budget Validation**: Rejects unrealistic budgets (e.g., "Laptop under Γé╣500")

**Data Source Restrictions**: Only uses approved sources (Amazon, Best Buy, Newegg, manufacturer websites). Rejects unknown or unverified review platforms.

**Recommendation Guardrails**:
- Avoid biased recommendations
- Explain ranking logic
- Disclose uncertainty when confidence is low
- Prevent hallucinated specifications
- Provide evidence-based recommendations only

---

### The Full Orchestration Function

Everything wired together in one function:

```python
def run_product_recommendation(query: str) -> RecommendationOutput:
    """Run ALL 4 agents in sequence. Returns final recommendation."""
    
    # Agent 0: Discovery
    ctx = search_products(f"{query} top models India {date.today().year}", 10)
    disc = discovery_chain.invoke({"user_query": query, "search_context": ctx})
    
    # Agent 1: Specs
    pl = "\n".join(f"- {p.brand} {p.model}" for p in disc.products)
    rd = ""
    for p in disc.products:
        n = f"{p.brand} {p.model}"
        rd += f"\n--- {n} ---\n{search_specs(n)}\n"
    sp = spec_chain.invoke({"product_list": pl, "research_data": rd})
    
    # Agent 2: Reviews
    rv = ""
    for p in disc.products:
        n = f"{p.brand} {p.model}"
        rv += f"\n--- {n} ---\n{search_reviews(n)}\n"
    rev = review_chain.invoke({"product_list": pl, "review_data": rv})
    
    # Agent 3: Recommendation
    ds = "\n".join(f"- {p.brand} {p.model}: {p.specs_hint}" for p in disc.products)
    ss = "\n".join(f"- {s.product_name}: CPU={s.cpu}, GPU={s.gpu}" for s in sp.specs)
    rs = "\n".join(f"- {r.product_name}: {r.overall_sentiment}" for r in rev.reviews)
    rec = recommendation_chain.invoke({
        "user_query": query, "discovery_data": ds,
        "spec_data": ss, "review_data": rs
    })
    return rec
```

**Constraint**: No LangGraph, no CrewAI, no AutoGen ΓÇö just pure Python functions connecting agents.

---

## How does it connect to other topics?

This project is the **capstone** ΓÇö every concept from the course comes together here:

| Course Concept | How It's Used Here |
|---|---|
| **LLMs** (01_genai_and_llm_fundamentals.md) | `ChatOpenAI(model="gpt-4o-mini")` powers every agent's brain |
| **Function Calling** (02_function_calling_and_tools.md) | Search tools (`search_products`, `search_specs`, `search_reviews`) are functions the pipeline calls |
| **Prompt Engineering** (03_prompt_engineering.md) | Each agent has a carefully crafted system prompt defining its role |
| **Pydantic / Structured Output** (04_pydantic_and_structured_output.md) | `llm.with_structured_output(Schema)` forces every agent to return clean data |
| **What Makes an Agent** (05_what_makes_an_agent.md) | Each agent = LLM + Tool + Prompt ΓÇö the fundamental agent pattern |
| **Multi-Agent Frameworks** (06_multi_agent_frameworks.md) | 4 agents orchestrated in a pipeline ΓÇö custom orchestration without frameworks |
| **Self-Reflection** (07_self_reflection_and_critique.md) | Confidence scores and trade-off analysis are a form of self-assessment |
| **Sentiment Analysis** (08_sentiment_analysis.md) | Agent 2 performs LLM-based sentiment analysis on reviews |
| **RAG** (10_rag_fundamentals.md) | The search + retrieve + generate pattern mirrors how each agent gathers external data |
| **Vector Databases** (11_vector_databases_and_faiss.md) | FAISS and embeddings power semantic retrieval of reviews and specs |

---

## Code Examples

### Example 1: Testing the Pipeline

```python
# Run the full pipeline
result = run_product_recommendation("Best laptops for AI development under 150000 INR")

# Access structured results
print("Top Pick:", result.top_pick)
print("Reason:", result.top_pick_reason)
print("Confidence:", result.confidence)

# Iterate through scores
for s in sorted(result.scores, key=lambda x: x.rank):
    print(f"#{s.rank} {s.product_name} ΓÇö {s.overall_score}/10")
```

### Example 2: Quick Sentiment Analysis Experiment

```python
class SentimentResult(BaseModel):
    text: str = Field(description="The original review text")
    sentiment: str = Field(description="Positive, Negative, or Mixed")
    confidence: float = Field(description="Confidence score 0.0 to 1.0")
    key_phrases: List[str] = Field(description="Key phrases that indicate sentiment")

sentiment_chain = llm.with_structured_output(SentimentResult)

result = sentiment_chain.invoke(
    "Analyze the sentiment: This laptop is incredible! RTX 4060 handles AI like a charm."
)
print(result.sentiment)       # "Positive"
print(result.confidence)      # 0.92
print(result.key_phrases)     # ["incredible", "handles AI like a charm"]
```

### Example 3: Different Query Types

```python
# Try different product categories
run_product_recommendation("Best headphones under 5000 INR")
run_product_recommendation("Best monitor for coding under 25000 INR")
run_product_recommendation("Best mobile under 20000 INR with good camera")
```

---

## Common Mistakes

1. **Not searching before invoking the LLM** ΓÇö The LLM can't browse the internet. You must call `search_products()` first, then pass the results to the LLM. Without search data, the LLM will hallucinate product names.

2. **Forgetting `with_structured_output()`** ΓÇö Without it, the LLM returns free-form text that you can't reliably parse. Always bind a Pydantic schema.

3. **Not passing data between agents** ΓÇö Agent 3 needs data from ALL previous agents. If you forget to format and pass `spec_result` or `review_result`, the recommendation will be based on incomplete information.

4. **Using the same search function for everything** ΓÇö The project uses 3 different search functions (`search_products`, `search_specs`, `search_reviews`) with different query templates. Each is tailored to find the right kind of information.

5. **Expecting Agent 3 to search** ΓÇö Agent 3 is a "reasoning-only" agent. It doesn't use any tools. It just analyzes data and makes decisions. This is an important design pattern: not every agent needs tools.

6. **Confusing the chain `|` operator** ΓÇö `prompt | llm.with_structured_output(Schema)` is LangChain's pipe operator (LCEL). It connects prompt ΓåÆ LLM. It's not a Python bitwise OR.

7. **Not handling search failures** ΓÇö The search function wraps calls in `try/except` and returns empty string on failure. Without this, one failed search crashes the entire pipeline.

8. **Hardcoding instead of using `.env`** ΓÇö Never put API keys directly in code. Always use `load_dotenv()` + `os.getenv("KEY_NAME")`.

---

## Practice Exercises

1. **Add a 5th agent**: Create a "Budget Advisor" agent that takes the recommendation output and generates a value-for-money analysis. Define a new Pydantic schema (`BudgetAdviceOutput`) with fields like `best_value_pick`, `premium_pick`, `budget_tip`. Wire it after Agent 3.

2. **Switch search tools**: Take the Tavily version and replace it with DuckDuckGo search. Compare the quality of recommendations. Which gives better results? Why?

3. **Add guardrail validation**: Write a `validate_query()` function that checks:
   - Is the category in the allowed list?
   - Is the budget realistic (not Γé╣500 for a laptop)?
   - Call it before Agent 0 runs.

4. **Change the scoring weights**: Modify Agent 3's prompt to use different weights (e.g., Reviews at 40% instead of 20%). Run the same query and compare how the ranking changes.

5. **Build a mini version**: Create a 2-agent pipeline for a simpler task ΓÇö e.g., "Restaurant Recommender" with Agent 0 (Discovery) and Agent 1 (Recommendation). Use the same 3-piece pattern (prompt + chain + invoke) for each agent.

---

## Key Takeaways

- Every agent = **Prompt + Chain + Invoke** ΓÇö learn this pattern and you can build any agent
- **Pydantic schemas** make the output of one agent consumable by the next
- **Custom orchestration** (plain Python) is simpler than frameworks for straightforward pipelines
- The LLM is the **brain**, search tools are the **eyes**, and Pydantic is the **contract**
- This project proves you don't need LangGraph or CrewAI for a working multi-agent system

---

**Repo location**: `mini_project_sprint/product_recommendation_agent.ipynb` (Tavily version), `mini_project_sprint/productrecomdagent.ipynb` (DuckDuckGo version with design docs)

**See also**: ALL other notes ΓÇö this is the capstone that connects everything:
- 01_genai_and_llm_fundamentals.md ΓåÆ 02_function_calling_and_tools.md ΓåÆ 03_prompt_engineering.md ΓåÆ 04_pydantic_and_structured_output.md ΓåÆ 05_what_makes_an_agent.md ΓåÆ 06_multi_agent_frameworks.md ΓåÆ 07_self_reflection_and_critique.md ΓåÆ 08_sentiment_analysis.md ΓåÆ 10_rag_fundamentals.md ΓåÆ 11_vector_databases_and_faiss.md ΓåÆ **09_building_product_recommendation_agent.md** (you are here)
