# Prompt Engineering

## What is it?
Prompt engineering is the art and science of crafting instructions (prompts) that guide a Large Language Model to produce the exact output you want — the right content, in the right format, at the right quality level.

---

## Why does it matter?
The same LLM with a bad prompt gives garbage; with a good prompt, it gives gold. You don't need a better model — you need a better prompt. In production AI systems, prompt engineering is often the highest-leverage skill: it's free (no retraining), fast (iterate in seconds), and determines whether your agent actually works or just rambles. Every agent, chatbot, and AI pipeline you build starts with a prompt.

---

## How does it work? (Under the Hood)

### The Prompt → Token → Probability Pipeline

```
Your Prompt (text)
    │
    ▼
┌─────────────────────────────┐
│  TOKENIZER                  │
│  Splits text into tokens    │
│  "What is GenAI?" → [What]  │
│  [is] [Gen] [AI] [?]       │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  TRANSFORMER MODEL          │
│  Processes all tokens with  │
│  self-attention              │
│  Assigns probability to     │
│  EVERY possible next token  │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  SAMPLING / DECODING        │
│                             │
│  temperature=0 → Greedy     │
│    (always pick highest     │
│     probability = "Blue")   │
│                             │
│  temperature=1 → Creative   │
│    (sample randomly from    │
│     probability distribution│
│     = might pick "cloudy")  │
│                             │
│  Top-K: only consider top   │
│    K most likely tokens     │
└──────────────┬──────────────┘
               │
               ▼
         Generated Text
```

### Why prompts matter at the math level
The LLM predicts: `P(next_token | all_previous_tokens)`. Your prompt IS the "all_previous_tokens". A better prompt shifts the probability distribution toward the tokens you actually want.

Think of it this way:
- Prompt: "sky" → next token probabilities: Blue(0.9), cloudy(0.6), falling(0.01)
- Prompt: "The stormy sky" → next token probabilities: darkened(0.8), thundered(0.7), Blue(0.2)

Same model, different prompt, completely different output.

---

## Types of Prompting

### 1. Zero-Shot Prompting
Just ask directly. No examples. The model relies entirely on its training.

```python
# From 4_Prompt_Essentials/Prompting_Introduction.ipynb
prompt = "What is genAI?"  # Zero shot prompting
response = llm.invoke(prompt)
```

**When to use**: Simple, well-known tasks (summarization, translation, basic Q&A).
**Limitation**: The model guesses the output format — you might get bullet points one time and paragraphs the next.

### 2. Instruction Prompting (Role + Context + Format)
Give the model a role, context about the audience, and a specific output format. This is the **ICOF framework** used in the repo:

```
#Instruction   → WHO is the model? What's its job?
#Context       → WHO is the audience? What's the situation?
#Input         → WHAT data to process?
#Output Format → HOW should the answer look?
```

```python
# From 4_Prompt_Essentials/Prompting_Introduction.ipynb
topic = "genai"

prompt = f"""
#Instruction
You are an technical AI assistant in explaining the AI concepts

#Context
You are required to explain the concept to an advanced IT professional

#Topic
{topic}

#Ouput Format
Output to be in 5 bullet points
"""

response = llm.invoke(prompt)
content = response.content
```

**Key insight**: Just adding "Output to be in 5 bullet points" changed the LLM from writing essays to giving concise bullets. Format constraints are incredibly powerful.

### 3. Few-Shot Prompting
Provide examples of input→output pairs BEFORE the actual question. The model learns the pattern from your examples.

```python
# From 4_Prompt_Essentials/Prompting_Introduction.ipynb — sentiment analysis
review = "This moview was quite good"
prompt = f"""
#Instruction
You are an AI assistant with the task of predicting the sentiment of the given movie review

#Context
Review should be labelled based on the sentiment

#Input
Movie Review : {review}

#Examples

Review : "I love this movie"
sentiment : "Pos"

Review : "This movie was okay"
setiment : "Neutral"

Review : "This movie was pathetic"
sentiment : "Neg"

#Ouput Format
The output to be returned as a JSON with key as "sentiment"
"""

response = llm.invoke(prompt)
response.content  # '{"sentiment":"Pos"}'
```

**Why it works**: Without examples, the model returned `"positive"`. With examples showing `"Pos"`, `"Neutral"`, `"Neg"`, it returned `"Pos"` — matching YOUR label format exactly.

### 4. Few-Shot with Format Guidance (Hashtag Example)

```python
# From 4_Prompt_Essentials/Prompting_Introduction.ipynb
sentence = "AI is transforming the world"
prompt = f"""
#Instruction
You are an AI assistant with the task of converting sentences into hashtag

#Context
Convert the given sentence into hashtags, keep in mind the hashtags
should be exact word from the sentence and the hashtag should only have one word

#Input
Sentence : {sentence}

#Few shot examples
Sentence : "Machine learning is powerful
Hastags : ["#MachineLearning" ,"#Powerful"]

Sentence : "AI is draining the water resources around the world
Hastags : ["AI", "#WaterResources", "#World", "#draining"]

#Ouput Format
The output to be returned as a JSON with key as "hashtags" and the value as list of hastags
"""
```

**Without few-shot**: Model returned `["#AI", "#is", "#transforming", "#the", "#world"]` — included stop words.
**With few-shot**: Model returned `["#AI", "#Transforming", "#World"]` — learned to skip unimportant words from the examples.

### 5. Chain-of-Thought (CoT) Prompting
Ask the model to "think step by step" before giving the final answer. This dramatically improves reasoning on complex tasks.

```python
# General pattern (not in repo but builds on the same ICOF structure)
prompt = """
#Instruction
You are a math tutor. Solve the problem step by step.

#Input
If a laptop costs ₹1,20,000 and has a 15% discount, what is the final price?

#Output Format
Show each step, then the final answer.
"""
```

**Why it works**: Forces the model to allocate more "thinking tokens" before jumping to an answer. Without CoT, models often skip steps and get wrong answers on math/logic problems.

---

## The Prompt Template Structure (ICOF)

Every good prompt follows this 4-part structure:

```
┌─────────────────────────────────────────┐
│  INSTRUCTION                            │
│  "You are a [role] expert in [domain]"  │
│  → Defines WHO the model is             │
├─────────────────────────────────────────┤
│  CONTEXT                                │
│  "Explain to [audience] for [purpose]"  │
│  → Defines the SITUATION                │
├─────────────────────────────────────────┤
│  INPUT                                  │
│  "Here is the data: {variable}"         │
│  → The actual DATA to process           │
├─────────────────────────────────────────┤
│  OUTPUT FORMAT                          │
│  "Return as JSON with keys..."          │
│  → Defines the SHAPE of the answer      │
└─────────────────────────────────────────┘
```

The more specific each section, the better the output.

---

## LangChain ChatPromptTemplate

When building real applications, you don't write prompts as raw f-strings. LangChain's `ChatPromptTemplate` gives you reusable, parameterized prompts with system/human message separation.

### Basic Pattern

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

# Define the template with {placeholders}
discovery_prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are a product discovery expert.
Analyze the user query and search results. Extract up to 5 real product models.
Ignore generic articles — only include specific products with brand and model.
"""),
    ("human", "User Query:\n{user_query}\n\nSearch Results:\n{search_context}")
])

# Connect to LLM (the pipe | operator = "chain")
chain = discovery_prompt | llm

# Run it — variables get filled in automatically
result = chain.invoke({
    "user_query": "Best laptops for AI development",
    "search_context": "...search results here..."
})
```

### Why system vs. human messages?
- **System message**: Sets the model's personality, role, and rules. Persists across the conversation. This is where agent behavior is defined.
- **Human message**: The actual user input for this specific request. Contains the data/question.

### How prompts connect to agents
In the mini project, EVERY agent is built with this exact 3-line pattern:

```python
# 1. Define prompt (with system personality + human data template)
prompt = ChatPromptTemplate.from_messages([...])

# 2. Chain = prompt | LLM (optionally with structured output)
chain = prompt | llm.with_structured_output(MySchema)

# 3. Run
result = chain.invoke({"var1": "value1", "var2": "value2"})
```

The system prompt IS the agent's personality. Change the system prompt = different agent.

---

## Temperature: The Creativity Dial

```python
# From 4_Prompt_Essentials/Prompting_Introduction.ipynb
llm1 = ChatOpenAI(model="gpt-5.4", temperature=0)  # Deterministic / Greedy
llm2 = ChatOpenAI(model="gpt-5.4", temperature=1)  # Creative / Random
```

- **temperature=0**: Always picks the highest-probability token. Same input = same output every time. Use for: factual tasks, data extraction, structured output.
- **temperature=1**: Samples randomly from the probability distribution. More creative, less predictable. Use for: brainstorming, creative writing, diverse suggestions.
- **temperature=0.3**: Slightly creative but mostly focused. The mini project uses this — good balance for product analysis.

---

## How does it connect to other topics?

- **See: `01_genai_and_llm_fundamentals.md`** — Prompts are how you communicate with LLMs. Understanding tokenization and next-token prediction explains WHY prompt wording matters.
- **See: `05_what_makes_an_agent.md`** — Agent system prompts define agent personality and behavior. The `ChatPromptTemplate` system message IS the agent's "brain."
- **See: `04_pydantic_and_structured_output.md`** — Output Format section of ICOF can be replaced entirely by Pydantic schemas via `with_structured_output()`. Much more reliable than asking for JSON in the prompt.
- **See: `07_self_reflection_and_critique.md`** — Self-reflection uses specialized prompts for the generator, critic, and refiner roles. Each role has a carefully crafted system message (see `5_Self_Reflection/prompts.py`).

---

## Code Examples

### Example 1: Zero-shot sentiment (unpredictable format)

```python
review = "This movie was okay"
prompt = f"""
#Instruction
You are an AI assistant with the task of predicting the sentiment

#Context
Review should be labelled based on the sentiment

#Input
Movie Review : {review}

#Ouput Format
The output to be returned as a JSON with key as "sentiment"
"""
response = llm.invoke(prompt)
response.content  # '{"sentiment":"neutral"}'
```

### Example 2: Few-shot sentiment (controlled format)

```python
# Adding examples forces the model to use YOUR label format
prompt = f"""
...
#Examples
Review : "I love this movie"  → sentiment : "Pos"
Review : "This movie was okay" → sentiment : "Neutral"
Review : "This movie was pathetic" → sentiment : "Neg"
...
"""
response.content  # '{"sentiment":"Pos"}' — matches YOUR format!
```

### Example 3: Reusable LLM helper with system/human separation

```python
from llm.llm import get_llm, invoke_llm
import os

llm = get_llm(os.getenv("OPENAI_API_KEY"))
response = invoke_llm("You are an AI Assistant", "What is GenAI", llm)
# System message and human message cleanly separated
```

### Example 4: Agent prompt with ChatPromptTemplate (from mini project)

```python
from langchain_core.prompts import ChatPromptTemplate

spec_prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are a hardware spec expert. Extract: CPU, GPU, RAM, Storage,
Display, Battery, Weight, Price.
If not found, write "Not available". Add a one-line AI/ML summary.
"""),
    ("human", "Products:\n{product_list}\n\nResearch Data:\n{research_data}")
])
spec_chain = spec_prompt | llm.with_structured_output(SpecificationOutput)
```

---

## Common Mistakes

1. **Vague instructions**: "Tell me about laptops" → rambling essay. Fix: "List the top 5 laptops for AI development under ₹1,50,000 with GPU specs."

2. **No output format**: You get a different format every time — sometimes bullet points, sometimes paragraphs, sometimes JSON. Always specify format.

3. **Too many instructions at once**: Cramming 10 tasks into one prompt confuses the model. Split into focused prompts (this is exactly why the mini project uses 4 separate agents).

4. **Ignoring temperature**: Using temperature=1 for data extraction gets inconsistent results. Use temperature=0 for structured/factual tasks.

5. **Not using few-shot when needed**: If the model keeps getting the format wrong, add 2-3 examples. Few-shot fixes most format issues.

6. **Forgetting the system message**: Putting everything in the human message. The system message sets persistent behavior rules — use it.

7. **Prompt injection vulnerability**: Not sanitizing user input in f-strings. A malicious user can override your instructions. In production, validate inputs.

---

## Practice Exercises

1. **Zero-shot vs. Few-shot comparison**: Take the sentiment analysis example from the notebook. Run the same review with zero-shot (no examples) and few-shot (3 examples). Compare: does the output format match? Which is more consistent?

2. **ICOF framework practice**: Pick any task (email summarizer, code reviewer, recipe generator). Write a prompt using all 4 parts: Instruction, Context, Input, Output Format. Test it with `llm.invoke()` and iterate until the output is exactly what you want.

3. **Temperature experiment**: Use the code from the notebook to invoke the same prompt with `temperature=0` and `temperature=1` five times each. Count how many unique responses you get at each temperature. What does this tell you?

4. **Build a ChatPromptTemplate**: Convert one of your f-string prompts into a `ChatPromptTemplate.from_messages()` with system and human messages. Chain it with `| llm` and invoke it. Then try adding `with_structured_output()` with a simple Pydantic model (see `04_pydantic_and_structured_output.md`).

5. **Chain-of-Thought experiment**: Ask the model "What is 17 × 24?" with a simple prompt, then ask again with "Think step by step before answering." Compare the accuracy. Try with a harder math problem to see the difference more clearly.
