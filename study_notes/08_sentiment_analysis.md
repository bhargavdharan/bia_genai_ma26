# Sentiment Analysis

**What is it?**
Sentiment analysis is the process of determining whether a piece of text expresses a positive, negative, or neutral opinion — and optionally extracting *why* (specific pros, cons, emotions, and aspect-level judgments).

---

## Why Does It Matter?

Every day, millions of product reviews, tweets, support tickets, and forum posts are written. No human can read them all, but businesses need to know: are customers happy or angry? Which specific features do they love or hate? Sentiment analysis automates this at scale. It powers Amazon's review summaries, brand monitoring dashboards, stock trading signals based on news tone, and — in our project — it lets an AI agent analyze hundreds of product reviews in seconds and produce structured pros/cons for a recommendation engine.

---

## How Does It Work? (Under the Hood)

### Three Approaches to Sentiment Analysis

```
┌──────────────────────────────────────────────────────────────────┐
│                  SENTIMENT ANALYSIS APPROACHES                   │
├──────────────────┬───────────────────┬───────────────────────────┤
│  1. RULE-BASED   │  2. ML CLASSIF.   │  3. LLM-BASED (ours)     │
│                  │                   │                           │
│  Word lists:     │  Train on labeled │  Ask the LLM directly:   │
│  positive_words  │  data:            │  "Analyze this review"   │
│  = [great, love] │  - Naive Bayes    │                           │
│  negative_words  │  - SVM            │  ✓ Handles sarcasm       │
│  = [bad, awful]  │  - Random Forest  │  ✓ Understands context   │
│                  │                   │  ✓ No training data       │
│  Count pos vs    │  Needs 1000s of   │  ✓ Extracts WHY          │
│  neg words       │  labeled examples │    (pros, cons, issues)  │
│                  │                   │                           │
│  ✗ Fails on:     │  ✗ Needs labeled  │  ✗ Costs per API call    │
│    "not bad"     │    training data  │  ✗ Slower than rules     │
│    sarcasm       │  ✗ Domain-specific│  ✗ Non-deterministic     │
│    context       │  ✗ Binary only    │                           │
└──────────────────┴───────────────────┴───────────────────────────┘
```

### Approach 1: Rule-Based (the old way)

Maintain a dictionary of positive and negative words. Count occurrences in the text.

```python
positive_words = {"great", "excellent", "love", "amazing", "fast", "beautiful"}
negative_words = {"terrible", "awful", "hate", "slow", "broken", "worst"}

def rule_based_sentiment(text):
    words = text.lower().split()
    pos = sum(1 for w in words if w in positive_words)
    neg = sum(1 for w in words if w in negative_words)
    if pos > neg: return "Positive"
    if neg > pos: return "Negative"
    return "Neutral"
```

**Why it fails:**
- `"Not bad at all"` → sees "bad" → says Negative (actually Positive!)
- `"Oh great, another laptop that overheats"` → sees "great" → says Positive (it's sarcasm!)
- `"The battery is not as terrible as I expected"` → sees "terrible" → Negative (it's actually a mild positive)

### Approach 2: ML Classification (traditional ML)

Train a classifier on labeled examples:

```
"Amazing laptop, love it!"    → Positive
"Worst purchase ever"         → Negative
"It's okay for the price"     → Neutral
```

Better than rules — the model learns patterns, not just individual words. But you need thousands of labeled examples, and the model is domain-specific (a model trained on movie reviews won't work well on laptop reviews).

### Approach 3: LLM-Based (what we use)

Just ask the LLM. It already understands language, sarcasm, and context from pre-training on billions of texts.

```
Traditional:   "Is this positive or negative?"  → "Positive"

LLM-Based:     "Analyze this review"            → {
                                                     sentiment: "Positive",
                                                     rating: "4.2/5",
                                                     pros: ["fast", "good display"],
                                                     cons: ["heavy", "loud fan"]
                                                   }
```

The LLM approach is strictly superior for our use case because:
1. **No training data needed** — works out of the box
2. **Handles context and sarcasm** — understands "not bad" is positive
3. **Extracts structured reasons** — not just positive/negative, but *why*
4. **Multi-aspect** — can judge different features separately

---

## Types of Sentiment Analysis

Understanding the different granularity levels helps you pick the right approach for your use case:

### 1. Binary Sentiment
The simplest form — just positive or negative.

```
"Great laptop!"                → Positive
"Terrible experience."         → Negative
```

Use case: Quick filtering (show me only positive reviews).

### 2. Ternary Sentiment
Adds a neutral/mixed category. This is what our project's `overall_sentiment` field uses.

```
"Amazing performance!"         → Positive
"It's okay, nothing special."  → Mixed
"Do not buy this."             → Negative
```

Use case: Sorting reviews into three buckets for summary dashboards.

### 3. Fine-Grained (Star Rating)
Predict a 1-5 or 1-10 score. Our project's `average_rating` field does this.

```
"Incredible in every way!"     → 5/5
"Good but has some flaws."     → 3.5/5
"Broke after 2 weeks."         → 1/5
```

Use case: Estimating a product's overall quality score from unstructured text.

### 4. Aspect-Based Sentiment
Sentiment *per feature* of a product. This is the most useful type for product comparison. Our project captures this via `pros` and `cons` lists.

```
Review: "Display is gorgeous but the battery barely lasts 3 hours."

Aspect breakdown:
  Display  → Positive
  Battery  → Negative
```

Use case: "Show me laptops where battery sentiment is positive."

### 5. Emotion Detection
Goes beyond positive/negative to identify specific emotions.

```
"I'm so frustrated with the support team!" → Frustrated, Angry
"This phone exceeded all my expectations!" → Excited, Happy
```

Use case: Customer support routing (send angry customers to senior agents).

### Our Project Combines Multiple Types

The `ProductReview` schema in the project uses a hybrid approach:

```
┌─────────────────────────────────────────────┐
│         ProductReview Schema                │
│                                             │
│  overall_sentiment: "Positive"   ← Ternary  │
│  average_rating: "4.3"          ← Fine-grain│
│  pros: ["Powerful GPU", ...]    ← Aspect +  │
│  cons: ["Heavy", "Loud fans"]   ← Aspect -  │
│  common_issues: ["Overheating"] ← Problems  │
│  expert_opinion: "Great for..." ← Summary   │
└─────────────────────────────────────────────┘
```

This gives us a **complete picture** — the overall feeling, a numeric score, what specifically is good, what's bad, and what recurring problems exist.

---

## How Sentiment Analysis Works in Our Product Recommendation Project

### The Pipeline

Agent 2 (the Review Analysis Agent) performs sentiment analysis as part of the 4-agent product recommendation system:

```
Agent 0 (Discovery)
    │ finds 5 products
    ▼
Agent 1 (Specifications)
    │ gets CPU/GPU/RAM/etc.
    ▼
Agent 2 (Review Analysis)  ◄── THIS IS WHERE SENTIMENT ANALYSIS HAPPENS
    │
    │  For EACH product:
    │    1. Search web: "{product} review pros cons 2026"
    │    2. Collect raw review text from multiple sources
    │    3. Feed ALL review text to LLM
    │    4. LLM returns ProductReview (structured sentiment analysis)
    │
    │ outputs sentiment + pros + cons for each product
    ▼
Agent 3 (Recommendation)
    │ uses review scores (20% weight) in final ranking
    ▼
Final Report
```

### The Pydantic Schema — `ProductReview`

This is the schema that forces the LLM to return structured sentiment analysis:

```python
from pydantic import BaseModel, Field
from typing import List

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

Each field serves a purpose:
- `overall_sentiment` — ternary classification for quick sorting
- `average_rating` — fine-grained numeric score for ranking
- `pros` / `cons` — aspect-based extraction for detailed comparison
- `common_issues` — recurring problems that might be dealbreakers
- `expert_opinion` — summarizes what professional reviewers think

### The Review Agent Code

Here's how Agent 2 is built and run in the project notebook:

```python
# BUILD: Review Agent prompt
review_prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are a review analysis expert. For each product determine:
sentiment (Positive/Mixed/Negative), rating/5, pros, cons, issues, expert opinion.
Base analysis on provided data. If limited, note it.
"""),
    ("human", "Products:\n{product_list}\n\nReview Data:\n{review_data}")
])

# BUILD: Chain = prompt → LLM with structured output
review_chain = review_prompt | llm.with_structured_output(ReviewOutput)
```

Running it:

```python
# Search reviews for each product
all_reviews = ""
for p in discovery_result.products:
    name = f"{p.brand} {p.model}"
    print(f"  Searching reviews: {name}")
    all_reviews += f"\n--- {name} ---\n{search_reviews(name)}\n"

# LLM analyzes all reviews at once → returns structured output
review_result = review_chain.invoke({
    "product_list": product_list_str,
    "review_data": all_reviews
})

# Display results
for r in review_result.reviews:
    print(f"\n{r.product_name} — {r.overall_sentiment} ({r.average_rating}/5)")
    print(f"  Pros: {', '.join(r.pros)}")
    print(f"  Cons: {', '.join(r.cons)}")
    print(f"  Issues: {', '.join(r.common_issues)}")
    print(f"  Expert: {r.expert_opinion}")
```

### The Sentiment Experiment

The notebook also includes a standalone sentiment analysis experiment (Part 5.5) that teaches the concept before applying it in Agent 2:

```python
# Simple sentiment schema for experimentation
class SentimentResult(BaseModel):
    text: str = Field(description="The original review text")
    sentiment: str = Field(description="Positive, Negative, or Mixed")
    confidence: float = Field(description="Confidence score 0.0 to 1.0")
    key_phrases: List[str] = Field(description="Key phrases that indicate sentiment")

# Create a sentiment chain — no prompt template needed, just structured output
sentiment_chain = llm.with_structured_output(SentimentResult)

# Test with sample reviews
sample_reviews = [
    "This laptop is incredible! The RTX 4060 handles AI workloads like a charm. "
    "Battery could be better though.",
    "Terrible experience. Screen flickered after 2 weeks. Customer support was unhelpful.",
    "It's decent for the price. Nothing extraordinary but gets the job done for basic ML tasks.",
]

for review in sample_reviews:
    result = sentiment_chain.invoke(
        f"Analyze the sentiment of this review:\n\n{review}"
    )
    print(f"Sentiment  : {result.sentiment}")
    print(f"Confidence : {result.confidence}")
    print(f"Key Phrases: {result.key_phrases}")
```

Notice the pattern: `llm.with_structured_output(SentimentResult)` forces the LLM to fill in every field of the Pydantic schema. No parsing, no regex, no post-processing needed.

### How Review Scores Feed Into Recommendations

Agent 3 (Recommendation) uses the review data as one of four scoring dimensions:

```python
class ProductScore(BaseModel):
    product_name: str = Field(description="Full product name")
    performance_score: float = Field(description="Hardware capability (0-10)")
    value_score: float = Field(description="Value for money (0-10)")
    review_score: float = Field(description="User satisfaction (0-10)")      # ← FROM SENTIMENT
    ai_readiness_score: float = Field(description="AI/ML suitability (0-10)")
    overall_score: float = Field(description="Weighted overall (0-10)")
    rank: int = Field(description="Final rank (1 = best)")
```

Scoring weights: Performance (30%) + Value (25%) + **Reviews (20%)** + AI Readiness (25%)

The review sentiment directly influences 20% of the final product score. A product with great specs but "Negative" sentiment and many `common_issues` will score lower than one with "Positive" sentiment and strong pros.

---

## How Does It Connect to Other Topics?

- **See: `04_pydantic_and_structured_output.md`** — `ProductReview`, `SentimentResult`, and `ReviewOutput` are all Pydantic schemas. The `Field(description=...)` pattern tells the LLM what to put in each field. Without Pydantic, you'd get unpredictable free-text.
- **See: `07_self_reflection_and_critique.md`** — The critique loop uses a similar evaluation pattern — scoring content against criteria and extracting structured feedback. The critic is doing "sentiment analysis" on draft quality.
- **See: `05_what_makes_an_agent.md`** — Agent 2 is built using the `prompt | llm.with_structured_output(Schema)` chain pattern. It uses a search tool (Tavily) to gather review data before analysis.
- **See: `09_building_product_recommendation_agent.md`** — Sentiment analysis is one piece of the full 4-agent pipeline. The review data flows into Agent 3 for final scoring.

---

## Common Mistakes

1. **Treating sentiment as just positive/negative** — Binary sentiment misses nuance. "Good laptop but terrible battery" is Mixed, not Positive. Always use at least ternary classification, and prefer aspect-based extraction when comparing products.

2. **Not handling sarcasm** — Rule-based and simple ML approaches fail on sarcasm. `"Oh wonderful, it crashed again"` contains "wonderful" but is clearly negative. LLM-based approaches handle this naturally, which is why we use them.

3. **Analyzing reviews one at a time instead of in batch** — The project sends ALL reviews for ALL products in a single LLM call. This is cheaper (fewer API calls) and gives the LLM context to compare products against each other. Sending one review at a time loses this comparative context.

4. **Ignoring `common_issues`** — Pros and cons capture what people like and dislike, but `common_issues` captures *recurring* problems. A single complaint about overheating could be a fluke; if it appears in 10 reviews, it's a pattern. This field specifically asks for recurring problems.

5. **Using high temperature for sentiment analysis** — Sentiment should be consistent. If you analyze the same review twice, you should get the same result. Use `temperature=0.3` or lower. The repo uses `0.3` for the review chain.

6. **Not specifying "based on provided data"** — Without this instruction, the LLM might hallucinate reviews from its training data. The prompt says "Base analysis on provided data. If limited, note it." This prevents the LLM from making up pros/cons.

7. **Confusing sentiment with opinion mining** — Sentiment tells you the *polarity* (positive/negative). Opinion mining tells you *what specifically* is being talked about. Our project does both — `overall_sentiment` is polarity, while `pros`/`cons`/`common_issues` are opinion mining. Most beginners only implement polarity and miss the more useful structured extraction.

---

## Practice Exercises

1. **Test sarcasm handling** — Add sarcastic reviews to the sentiment experiment in the notebook:
   ```python
   "Oh great, another laptop that overheats. Just what I needed."
   "Sure, if you enjoy watching loading screens, this is the laptop for you."
   ```
   Does the LLM correctly identify these as negative? Compare with a rule-based approach.

2. **Add emotion detection** — Extend the `SentimentResult` schema with an `emotions: List[str]` field described as "Detected emotions (e.g., frustrated, excited, disappointed)". Run the experiment and see what emotions the LLM detects for each review.

3. **Build aspect-based sentiment** — Create a new Pydantic schema:
   ```python
   class AspectSentiment(BaseModel):
       aspect: str = Field(description="Product feature (e.g., Battery, Display)")
       sentiment: str = Field(description="Positive, Negative, or Neutral")
       evidence: str = Field(description="Quote from the review supporting this")
   ```
   Feed a long review and extract per-aspect sentiments.

4. **Compare approaches** — Implement the rule-based approach (word counting) and compare its output to the LLM approach on the same 10 reviews. Count how many times they disagree and who is correct.

5. **Try multilingual sentiment** — Test the LLM with reviews in Hinglish: `"Laptop bahut accha hai, but battery kharab hai"`. Does the LLM handle mixed-language reviews? What about pure Hindi or Tamil reviews? This is a real-world challenge for Indian product analysis.
