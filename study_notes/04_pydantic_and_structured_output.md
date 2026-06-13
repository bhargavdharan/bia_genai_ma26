# Pydantic and Structured Output

## What is it?
Pydantic is Python's data validation library that uses type hints to define strict data schemas — and when combined with LangChain's `with_structured_output()`, it forces LLMs to return data in an exact, predictable format every single time.

---

## Why does it matter?
Without structured output, an LLM returns free-form text. Ask "what laptops do you recommend?" three times and you'll get three different formats — sometimes bullet points, sometimes paragraphs, sometimes partial JSON. You can't write code that reliably processes unpredictable text. Pydantic solves this: you define the exact shape of data you want, and the LLM is constrained to fill it. This is the foundation of every production AI agent — without structured output, agents can't pass data to each other reliably.

---

## How does it work? (Under the Hood)

### The Problem: Unpredictable LLM Output

```
Ask LLM: "What are the best laptops?"

Response attempt 1: "ASUS ROG Strix is a great laptop..."       ← plain text
Response attempt 2: {"laptop": "ASUS"}                          ← random JSON
Response attempt 3: ["ASUS", "Lenovo"]                          ← just a list
Response attempt 4: "Here are my top picks:\n1. ASUS ROG..."    ← markdown list

Same question → 4 different formats → your code breaks every time
```

### The Solution Pipeline

```
Step 1: You define a Pydantic schema
         ┌──────────────────────────────────────┐
         │  class DiscoveredProduct(BaseModel):  │
         │      brand: str                       │
         │      model: str                       │
         │      reason: str                      │
         └───────────────┬──────────────────────┘
                         │
Step 2: LangChain converts it to JSON Schema
                         │
                         ▼
         ┌──────────────────────────────────────┐
         │  {                                    │
         │    "type": "object",                  │
         │    "properties": {                    │
         │      "brand": {                       │
         │        "type": "string",              │
         │        "description": "Product        │
         │         manufacturer"                 │
         │      },                               │
         │      "model": {"type": "string"},     │
         │      "reason": {"type": "string"}     │
         │    },                                 │
         │    "required": ["brand","model",      │
         │                  "reason"]            │
         │  }                                    │
         └───────────────┬──────────────────────┘
                         │
Step 3: JSON schema sent to LLM alongside your prompt
         (OpenAI calls this "function calling" / "tool use")
                         │
                         ▼
         ┌──────────────────────────────────────┐
         │  LLM is CONSTRAINED to generate      │
         │  tokens that form valid JSON          │
         │  matching the schema                  │
         │                                       │
         │  Output: {"brand": "ASUS",            │
         │           "model": "ROG Strix G16",   │
         │           "reason": "Best GPU for AI"}│
         └───────────────┬──────────────────────┘
                         │
Step 4: LangChain validates response against Pydantic
                         │
                         ▼
         ┌──────────────────────────────────────┐
         │  You get a Python object:             │
         │  result.brand  → "ASUS"               │
         │  result.model  → "ROG Strix G16"      │
         │  result.reason → "Best GPU for AI"    │
         └──────────────────────────────────────┘
```

### The key insight
The LLM isn't "parsing" its output into JSON after generating it. The JSON schema is sent to the model AS PART OF THE PROMPT, and the model's token-by-token generation is constrained to only produce valid JSON matching that schema. It's like giving someone a form to fill out instead of asking them to write a free-form essay.

---

## Pydantic Basics

### BaseModel — Your Data Blueprint

```python
from pydantic import BaseModel, Field

class DiscoveredProduct(BaseModel):
    brand: str = Field(description="Product manufacturer (e.g., ASUS, Lenovo)")
    model: str = Field(description="Product model name (e.g., ROG Strix G16)")
    reason: str = Field(description="Why this product matches the user query")
```

Every field has:
- **A name**: `brand`, `model`, `reason`
- **A type**: `str`, `int`, `float`, `List[str]`, `bool`, etc.
- **A description** (via `Field`): This is the LLM's instruction for what to put in each field

### Why `Field(description=...)` Matters
The description is NOT just documentation for humans — it's literally the instruction the LLM reads when deciding what to put in that field. Compare:

```python
# BAD — LLM has no guidance, might put anything
class Product(BaseModel):
    x: str
    y: str

# GOOD — LLM knows exactly what each field expects
class Product(BaseModel):
    brand: str = Field(description="Product manufacturer (e.g., ASUS, Lenovo)")
    model: str = Field(description="Product model name (e.g., ROG Strix G16)")
```

Think of `description` as a mini-prompt for each field. The more specific, the better the LLM fills it.

### Creating Pydantic Objects Manually

```python
# From mini_project_sprint/product_recommendation_agent.ipynb
test = DiscoveredProduct(
    brand="ASUS",
    model="ROG Strix G16",
    reason="Great for AI"
)

print(test)          # brand='ASUS' model='ROG Strix G16' reason='Great for AI'
print(test.brand)    # ASUS (dot access!)
print(test.model_dump())  # {'brand': 'ASUS', 'model': 'ROG Strix G16', 'reason': 'Great for AI'}
```

### `model_dump()` — Convert to Dictionary
When you need to pass Pydantic data to JSON, APIs, or other functions:

```python
product = DiscoveredProduct(brand="ASUS", model="ROG Strix G16", reason="Best GPU")

# Convert to plain Python dict
data = product.model_dump()
# {'brand': 'ASUS', 'model': 'ROG Strix G16', 'reason': 'Best GPU'}

# Now you can serialize to JSON, save to file, pass between agents, etc.
import json
json_str = json.dumps(data)
```

### Validation — What Happens When Data Doesn't Match

```python
# Missing required field → ValidationError!
try:
    bad = DiscoveredProduct(brand="ASUS")  # Missing 'model' and 'reason'
except Exception as e:
    print(e)
    # 2 validation errors for DiscoveredProduct
    # model
    #   Field required [type=missing]
    # reason
    #   Field required [type=missing]

# Wrong type → ValidationError!
try:
    bad = DiscoveredProduct(brand=123, model="ROG", reason="test")  # brand should be str
except Exception as e:
    print(e)  # Pydantic will actually coerce 123 to "123" for str fields
```

Pydantic v2 is strict about types. If you define `score: int` and the LLM returns `"8"` (string), Pydantic handles the coercion. But if the LLM returns `"excellent"` for an `int` field, it raises `ValidationError`.

---

## Nested Models and Lists

Real-world schemas are nested — a parent model contains lists of child models:

```python
from typing import List

# Child model — one product
class DiscoveredProduct(BaseModel):
    brand: str = Field(description="Product manufacturer (e.g., ASUS, Lenovo)")
    model: str = Field(description="Product model name (e.g., ROG Strix G16)")
    specs_hint: str = Field(description="Brief specs from search (e.g., RTX 4060, 16GB)")
    source: str = Field(description="URL where this product was found")
    reason: str = Field(description="Why this product matches the user query")

# Parent model — the full output containing a LIST of products
class DiscoveryOutput(BaseModel):
    category: str = Field(description="Detected category (e.g., Laptops)")
    budget: str = Field(description="Detected budget (e.g., Under 1,50,000)")
    use_case: str = Field(description="Detected use case (e.g., AI Development)")
    products: List[DiscoveredProduct] = Field(description="List of products (up to 5)")
```

The LLM returns a JSON object with a `products` array, and each item in the array matches `DiscoveredProduct`. LangChain + Pydantic validates the entire nested structure automatically.

---

## `with_structured_output()` — The Magic Method

This is where Pydantic connects to LLMs:

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

# Define prompt
discovery_prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are a product discovery expert.
Analyze the user query and search results. Extract up to 5 real product models.
"""),
    ("human", "User Query:\n{user_query}\n\nSearch Results:\n{search_context}")
])

# Chain with structured output — THIS is the key line
discovery_chain = discovery_prompt | llm.with_structured_output(DiscoveryOutput)

# Invoke — returns a DiscoveryOutput object, NOT raw text
result = discovery_chain.invoke({
    "user_query": "Best laptops for AI development",
    "search_context": "...search results..."
})

# Access data with dot notation
print(result.category)               # "Laptops"
print(result.budget)                  # "Under ₹1,50,000"
print(result.products[0].brand)      # "ASUS"
print(result.products[0].model)      # "ROG Strix G16"
print(type(result))                  # <class 'DiscoveryOutput'>
```

### What `with_structured_output()` does behind the scenes:
1. Takes your Pydantic model class
2. Calls `model.model_json_schema()` to get the JSON schema
3. Sends the schema to the LLM via OpenAI's function calling / tool use API
4. The LLM generates constrained JSON
5. LangChain parses the JSON back into a Pydantic object
6. Validates all fields — if something is wrong, raises an error

---

## Real Schemas from the Product Recommendation Project

### Agent 0: Discovery Agent — `DiscoveryOutput`

```python
# From mini_project_sprint/product_recommendation_agent.ipynb
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
```

### Agent 1: Specification Agent — `TechnicalSpecs`

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

### Agent 2: Review Analysis Agent — `ProductReview`

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

### Agent 3: Recommendation Agent — `ProductScore`

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

### Notice the pattern across all 4 agents:
1. **Child model** defines one item (one product, one spec, one review, one score)
2. **Parent model** wraps a `List[ChildModel]` plus metadata fields
3. Each agent's chain uses `llm.with_structured_output(ParentModel)`
4. Output from Agent N becomes input text for Agent N+1

---

## Advanced Pydantic: Validators and Constraints

The self-reflection module (`5_Self_Reflection/schemas.py`) shows more advanced Pydantic features:

### Literal Types — Restricting to specific values

```python
from typing import Literal

Severity = Literal["minor", "major", "blocking"]

class CritiqueIssue(BaseModel):
    criterion: str = Field(..., min_length=2)
    severity: Severity  # Can ONLY be "minor", "major", or "blocking"
    explanation: str = Field(..., min_length=10)
    revision_instruction: str = Field(..., min_length=10)
```

### Field Constraints — `ge`, `le`, `min_length`

```python
class CritiqueResult(BaseModel):
    score: int = Field(..., ge=1, le=10)     # Must be between 1 and 10
    passed: bool
    summary: str = Field(..., min_length=10)  # At least 10 characters
    strengths: list[str] = Field(default_factory=list)
    issues: list[CritiqueIssue] = Field(default_factory=list)
    reflection_memory: str = Field(..., min_length=10)
    revised_strategy: str = Field(..., min_length=10)
```

### Custom Validators — `@field_validator`

```python
from pydantic import field_validator

class CritiqueResult(BaseModel):
    # ... fields above ...

    @field_validator("strengths")
    @classmethod
    def limit_strengths(cls, strengths: list[str]) -> list[str]:
        """Keep output readable — max 4 strengths."""
        return strengths[:4]

    @field_validator("issues")
    @classmethod
    def limit_issues(cls, issues: list[CritiqueIssue]) -> list[CritiqueIssue]:
        """Keep the refiner focused — max 5 issues."""
        return issues[:5]
```

These validators run AFTER the LLM generates output. Even if the LLM returns 10 issues, the validator trims to 5. This is a safety net — you control the final output.

### Properties on Pydantic Models

```python
class IterationRecord(BaseModel):
    iteration: int = Field(..., ge=1)
    draft: str = Field(..., min_length=1)
    critique: CritiqueResult

    @property
    def score(self) -> int:
        """Quick access: record.score instead of record.critique.score"""
        return self.critique.score

    @property
    def passed(self) -> bool:
        """Quick access: record.passed instead of record.critique.passed"""
        return self.critique.passed
```

---

## How does it connect to other topics?

- **See: `03_prompt_engineering.md`** — The "Output Format" section of ICOF prompts is the manual version of what Pydantic does automatically. `with_structured_output()` replaces asking for JSON in the prompt text.
- **See: `05_what_makes_an_agent.md`** — Every agent in the mini project uses Pydantic schemas. The schemas define what data each agent produces and what the next agent receives.
- **See: `07_self_reflection_and_critique.md`** — The `CritiqueResult` and `CritiqueIssue` schemas (`5_Self_Reflection/schemas.py`) use advanced Pydantic features like `Literal` types, `field_validator`, and constraint parameters.
- **See: `09_building_product_recommendation_agent.md`** — All 4 agents (`DiscoveryOutput`, `SpecificationOutput`, `ReviewOutput`, `RecommendationOutput`) are built entirely on Pydantic schemas. The schemas ARE the contract between agents.

---

## Code Examples

### Example 1: Basic schema + LLM structured output

```python
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

class MovieSentiment(BaseModel):
    movie: str = Field(description="Name of the movie")
    sentiment: str = Field(description="Positive, Negative, or Neutral")
    confidence: float = Field(description="Confidence score 0.0 to 1.0")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
prompt = ChatPromptTemplate.from_messages([
    ("system", "Analyze the sentiment of movie reviews."),
    ("human", "Review: {review}")
])

chain = prompt | llm.with_structured_output(MovieSentiment)
result = chain.invoke({"review": "This movie was absolutely fantastic!"})

print(result.movie)       # The movie name
print(result.sentiment)   # "Positive"
print(result.confidence)  # 0.95
```

### Example 2: Inspecting schema fields

```python
# Check what fields a schema expects
print(DiscoveryOutput.model_fields.keys())
# dict_keys(['category', 'budget', 'use_case', 'products'])

# See the full JSON schema that gets sent to the LLM
import json
print(json.dumps(DiscoveryOutput.model_json_schema(), indent=2))
```

### Example 3: Data flow between agents using Pydantic

```python
# Agent 0 returns structured data
discovery_result = discovery_chain.invoke({...})

# Agent 1 uses Agent 0's output as input text
product_list_str = "\n".join(
    f"- {p.brand} {p.model}" for p in discovery_result.products
)

# Agent 1 returns its own structured data
spec_result = spec_chain.invoke({
    "product_list": product_list_str,
    "research_data": all_research
})

# Access nested data easily
for spec in spec_result.specs:
    print(f"{spec.product_name}: {spec.gpu}, {spec.ram}")
```

---

## Common Mistakes

1. **Forgetting `Field(description=...)`**: Without descriptions, the LLM guesses what to put in each field. Always add clear descriptions — they're mini-prompts.

2. **Using `str` for everything**: If a field is a number (like a score), use `float` or `int`. Pydantic validates types and the LLM generates appropriate values.

3. **Not handling `ValidationError`**: When the LLM returns bad data, `with_structured_output()` raises an error. Wrap calls in try/except in production code.

4. **Overly complex schemas**: Don't nest 5 levels deep. Keep schemas flat when possible. The LLM has a harder time filling deeply nested structures accurately.

5. **Confusing `model_dump()` with `dict()`**: In Pydantic v2, use `model_dump()`. The old `dict()` method is deprecated. Similarly, use `model_json_schema()` not `schema()`.

6. **Not testing schemas manually first**: Before connecting to an LLM, create test objects manually (like the `test = DiscoveredProduct(...)` example). This catches schema design issues early.

7. **Missing `List` import**: `products: List[DiscoveredProduct]` requires `from typing import List`. Without it, you get a confusing error. (In Python 3.9+, you can also use `list[DiscoveredProduct]`.)

---

## Practice Exercises

1. **Build a simple schema**: Create a `BookReview` Pydantic model with fields: `title` (str), `author` (str), `rating` (int, 1-5), `summary` (str), `recommend` (bool). Add `Field(description=...)` for each. Create a test object manually and print it, then try `model_dump()`.

2. **Connect to LLM**: Use your `BookReview` schema with `llm.with_structured_output(BookReview)`. Ask the LLM to review a real book. Verify you get a proper `BookReview` object back with dot access.

3. **Nested schema**: Create a `BookList` model that contains `books: List[BookReview]`. Ask the LLM to review 3 books at once. Access individual book ratings with `result.books[0].rating`.

4. **Validation experiment**: Create a Pydantic model with `score: int = Field(..., ge=1, le=10)`. Try creating an object with `score=15`. What error do you get? Now try with `score=7`. Then try connecting it to an LLM — does the LLM respect the constraint?

5. **Schema comparison**: Run the same prompt twice — once with `llm.invoke()` (plain text output) and once with `llm.with_structured_output(YourSchema)`. Compare: which is easier to use in code? Which is more consistent across multiple runs?
