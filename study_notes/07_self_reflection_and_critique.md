# Self-Reflection and Critique in AI Systems

**What is it?**
Self-reflection is a pattern where an AI system judges its own output against a set of criteria, identifies problems, and rewrites the output — repeating this loop until the result is good enough.

---

## Why Does It Matter?

LLMs generate text in a single pass — they don't naturally "think twice." The first draft is often generic, vague, or misses requirements. Self-reflection adds a quality control loop that mimics how humans revise their work: write a draft, review it critically, fix the weak spots, repeat. This is critical in production systems where you need consistent, high-quality outputs — think marketing copy, reports, code generation, or any task where "good enough on the first try" isn't reliable.

---

## How Does It Work? (Under the Hood)

### The Generate → Critique → Refine Loop

This is the core pattern. Three separate LLM calls play three different roles:

```
┌─────────────────────────────────────────────────────────┐
│                    ENTRY POINT                          │
│  Brief + Rubric + Constitution + (optional weak draft)  │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │  1. GENERATOR       │
              │  (or initial draft) │
              │  Creates first      │
              │  version of text    │
              └─────────┬───────────┘
                        │  draft
                        ▼
              ┌─────────────────────┐
              │  2. CRITIC          │◄────── Rubric
              │  Evaluates draft    │◄────── Constitution
              │  Returns JSON:      │◄────── Reflection Memory
              │  - score (1-10)     │
              │  - strengths        │
              │  - issues[]         │
              │  - reflection_memory│
              └─────────┬───────────┘
                        │  CritiqueResult
                        ▼
              ┌─────────────────────┐
              │  STOPPING RULE      │
              │  (Code, NOT LLM!)   │
              │                     │
              │  score >= pass_score │
              │  AND                │
              │  no blocking issues │
              └───┬─────────┬───────┘
                  │         │
             PASS │         │ FAIL
                  │         │
                  ▼         ▼
              ┌──────┐  ┌─────────────────────┐
              │ DONE │  │  3. REFINER          │
              │      │  │  Rewrites draft      │
              └──────┘  │  using critique      │
                        │  + constitution      │
                        │  + reflection memory │
                        └─────────┬────────────┘
                                  │  revised draft
                                  │
                                  ▼
                         (back to step 2)
                    Loop until PASS or max_iterations
```

### The Three Roles — Each is a Separate Prompt

**Generator** — writes the first draft. Gets a system prompt that says "you are a concise professional copywriter" and nothing else. No self-critique allowed here.

**Critic** — evaluates the draft against a rubric. Returns structured JSON with scores, issues, and a reflection memory. Uses `temperature=0.0` for deterministic evaluation and `json_mode=True` to force valid JSON output.

**Refiner** — rewrites the draft using the critique. Gets the original brief, the current draft, the critique JSON, the constitution, and all prior reflection memories. Uses `temperature=0.3` for focused but slightly creative output.

---

## Key Components Deep-Dive

### 1. Pydantic Schemas (from `5_Self_Reflection/schemas.py`)

The schemas enforce structure on the LLM's critique output. Without them, the critic might return free-text opinions that are impossible to parse programmatically.

```python
from pydantic import BaseModel, Field, field_validator
from typing import Literal

Severity = Literal["minor", "major", "blocking"]

class CritiqueIssue(BaseModel):
    """One concrete issue found by the evaluator."""
    criterion: str = Field(..., min_length=2)          # Which rubric criterion
    severity: Severity                                  # minor | major | blocking
    explanation: str = Field(..., min_length=10)        # What's wrong
    revision_instruction: str = Field(..., min_length=10)  # How to fix it

class CritiqueResult(BaseModel):
    """Structured evaluator output for a draft."""
    score: int = Field(..., ge=1, le=10)               # Overall score
    passed: bool                                        # Did it meet threshold?
    summary: str = Field(..., min_length=10)            # Short overall judgment
    strengths: list[str] = Field(default_factory=list)  # What to preserve
    issues: list[CritiqueIssue] = Field(default_factory=list)  # Problems to fix
    reflection_memory: str = Field(..., min_length=10)  # Lesson for next iteration
    revised_strategy: str = Field(..., min_length=10)   # Approach for rewrite

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

class IterationRecord(BaseModel):
    """One pass through the generate-critique-refine loop."""
    iteration: int = Field(..., ge=1)
    draft: str = Field(..., min_length=1)
    critique: CritiqueResult

    @property
    def score(self) -> int:
        return self.critique.score

    @property
    def passed(self) -> bool:
        return self.critique.passed
```

**Why validators matter:** `limit_strengths` and `limit_issues` prevent the LLM from dumping 20 issues that overwhelm the refiner. Keeping it to 5 issues forces focus on the highest-impact problems.

### 2. The Stopping Rule — Code Enforces This, Not the LLM

This is one of the most important design decisions in the repo. Look at this from `critic_loop.py`:

```python
def critique_draft(brief, draft, rubric, constitution, reflections=None, pass_score=8):
    messages = build_critic_messages(...)
    raw = call_chat_model(messages, temperature=0.0, max_tokens=900, json_mode=True)
    parsed = parse_json_object(raw)
    critique = CritiqueResult.model_validate(parsed)

    # The application owns the stopping rule. The model can recommend,
    # but code enforces.
    has_blocking_issue = any(issue.severity == "blocking" for issue in critique.issues)
    critique.passed = critique.score >= pass_score and not has_blocking_issue
    return critique
```

The LLM sets `passed` in its JSON, but the code **overwrites** it using its own logic. Why? Because LLMs are unreliable judges of their own thresholds — they might say "passed" when the score is 6 or "failed" when the score is 9. The code says: "I don't care what you think — if the score is ≥ 8 AND there are no blocking issues, then it passes."

### 3. Reflection Memory — Lessons That Carry Forward

Each iteration produces a `reflection_memory` string — a one-sentence lesson learned. These accumulate across iterations and are injected into both the critic and refiner prompts:

```python
# From run_refinement_loop() in critic_loop.py
reflections: list[str] = []

for iteration in range(1, max_iterations + 1):
    critique = critique_draft(..., reflections=reflections, ...)
    reflections.append(critique.reflection_memory)  # ← accumulates

    if critique.passed:
        break

    draft = refine_draft(..., reflections=reflections, ...)
```

Example reflections across iterations:
- Iteration 1: `"Avoid generic AI hype; tie every claim to a concrete program feature."`
- Iteration 2: `"Specific, credible benefits outperform generic AI enthusiasm for this audience."`

These are formatted into prompts as a bullet list:

```python
def _join_reflections(reflections):
    if not reflections:
        return "No prior reflections yet."
    return "\n".join(f"- {item}" for item in reflections if item.strip())
```

### 4. Constitutional AI — Principles That Guide Behavior

The constitution is a set of rules the AI must follow. From the repo's `data/constitution.md`:

```markdown
# Principles for the Critic

- Be accurate: do not invent guarantees, rankings, salaries, or outcomes.
- Be specific: prefer concrete benefits over generic enthusiasm.
- Be audience-aware: working professionals have limited time and low tolerance for hype.
- Be concise: every sentence should move the reader toward a decision.
- Be useful: critique should produce revision instructions, not vague opinions.
```

The constitution is injected into both the critic prompt and the refiner prompt. It's the "guardrails" — telling the LLM what kind of output is acceptable and what isn't.

### 5. Rubric-Based Evaluation

The rubric defines specific scoring criteria. From `data/rubric.md`:

```markdown
# Writing Quality Rubric

Score each draft from 1 to 10 using these criteria:

1. Audience fit — speaks to working professionals without hype.
2. Specificity — includes concrete program benefits, not vague claims.
3. Constraint following — respects word count, tone, and required elements.
4. Evidence quality — avoids unsupported promises or unrealistic outcomes.
5. Call to action — ends with one clear, practical next step.

Passing threshold for the classroom demo: 8/10.
```

Each `CritiqueIssue` references a specific `criterion` from this rubric, making the feedback actionable rather than vague.

---

## The Complete Code Flow

Here's the real entry point from `main.py`:

```python
from critic_loop import read_text, run_refinement_loop, print_iteration_trace, save_demo_outputs

def main():
    brief = read_text("data/writing_brief.txt")
    weak_draft = read_text("data/weak_draft.txt")
    rubric = read_text("data/rubric.md")
    constitution = read_text("data/constitution.md")

    final_draft, trace = run_refinement_loop(
        brief=brief,
        initial_draft=weak_draft,
        rubric=rubric,
        constitution=constitution,
        pass_score=8,
        max_iterations=3,
    )

    print_iteration_trace(trace)
    print("\nFINAL DRAFT\n")
    print(final_draft)
    save_demo_outputs(final_draft, trace)
```

And the core loop in `critic_loop.py`:

```python
def run_refinement_loop(brief, initial_draft, rubric, constitution,
                        pass_score=8, max_iterations=3):
    draft = initial_draft.strip() if initial_draft else generate_first_draft(brief)
    trace: list[IterationRecord] = []
    reflections: list[str] = []

    for iteration in range(1, max_iterations + 1):
        critique = critique_draft(
            brief=brief, draft=draft, rubric=rubric,
            constitution=constitution, reflections=reflections,
            pass_score=pass_score,
        )
        record = IterationRecord(iteration=iteration, draft=draft, critique=critique)
        trace.append(record)
        reflections.append(critique.reflection_memory)

        if critique.passed:
            break

        draft = refine_draft(
            brief=brief, draft=draft, critique=critique,
            constitution=constitution, reflections=reflections,
        )

    return draft, trace
```

### What a Typical Run Looks Like

Starting with a weak draft like:

> "Hi everyone, AI is changing the world and you should learn it as soon as possible. Our course will make you skilled in all AI tools..."

**Iteration 1:** Score 6/10 — "too generic for a skeptical audience"
- Issue: [major] Specificity — "broad claims like 'future ready' without explaining what the program builds"
- Issue: [major] Evidence quality — "vague career promises"
- Reflection: "Avoid generic AI hype; tie every claim to a concrete program feature."

**Iteration 2:** Score 9/10 — "specific, credible, well-matched to working professionals"
- Issue: [minor] Constraint following — "check word count"
- Reflection: "Specific, credible benefits outperform generic enthusiasm."
- **PASSED** ✓

---

## Prompt Design Details

### Critic Prompt — Forces JSON Output

```python
def build_critic_messages(brief, draft, rubric, constitution, reflections, pass_score):
    schema_instructions = """
Return ONLY valid JSON with this exact shape:
{
  "score": 1,
  "passed": false,
  "summary": "short overall judgment",
  "strengths": ["what is working"],
  "issues": [
    {
      "criterion": "rubric criterion name",
      "severity": "minor | major | blocking",
      "explanation": "what is wrong",
      "revision_instruction": "specific rewrite instruction"
    }
  ],
  "reflection_memory": "one reusable lesson for the next attempt",
  "revised_strategy": "the next rewrite strategy"
}
"""
    return [
        {"role": "system", "content":
            "You are a strict but constructive evaluator. Judge the draft against "
            "the brief, rubric, and principles. Do not praise vague text. "
            "A draft passes only when it earns at least the required score and "
            "has no blocking issue."},
        {"role": "user", "content":
            f"Original brief:\n{brief}\n\n"
            f"Rubric:\n{rubric}\n\n"
            f"Principles / constitution:\n{constitution}\n\n"
            f"Prior reflection memory:\n{_join_reflections(reflections)}\n\n"
            f"Passing score: {pass_score}/10\n\n"
            f"Draft to critique:\n{draft}\n\n"
            f"{schema_instructions}"},
    ]
```

### Refiner Prompt — Gets Critique as JSON

```python
def build_refiner_messages(brief, draft, critique_json, constitution, reflections):
    return [
        {"role": "system", "content":
            "You are a refinement editor. Rewrite the draft using the critique. "
            "Return only the revised draft, with no commentary."},
        {"role": "user", "content":
            f"Original brief:\n{brief}\n\n"
            f"Principles to preserve:\n{constitution}\n\n"
            f"Prior reflection memory:\n{_join_reflections(reflections)}\n\n"
            f"Current draft:\n{draft}\n\n"
            f"Structured critique:\n{critique_json}\n\n"
            "Rewrite the draft so it better satisfies the brief and rubric."},
    ]
```

Notice the refiner gets `critique_json` — the full structured critique serialized as JSON via `critique.model_dump_json(indent=2)`. This gives the refiner access to every issue, its severity, and the specific revision instruction.

---

## How Does It Connect to Other Topics?

- **See: `03_prompt_engineering.md`** — The generator, critic, and refiner each use carefully designed system prompts. The critic prompt includes JSON schema instructions (a form of structured prompting).
- **See: `04_pydantic_and_structured_output.md`** — `CritiqueResult`, `CritiqueIssue`, and `IterationRecord` are all Pydantic schemas with validators. The `json_mode=True` parameter forces the LLM to output valid JSON.
- **See: `05_what_makes_an_agent.md`** — The generate-critique-refine loop is a multi-step agent pattern where each "agent" (generator, critic, refiner) has a specific role.
- **See: `08_sentiment_analysis.md`** — The review agent in the product project uses a similar evaluation pattern — structured output for scoring and extracting pros/cons.

---

## Common Mistakes

1. **Trusting the LLM's `passed` field** — The LLM might say "passed=true" when the score is 5. Always override `passed` in code using your own threshold logic.

2. **Not using `json_mode=True`** — Without it, the LLM might wrap JSON in markdown code fences or add explanatory text before/after the JSON, breaking your parser.

3. **Too many issues per iteration** — If you let the LLM dump 15 issues, the refiner gets confused trying to fix everything at once. The repo caps it at 5 via `@field_validator`.

4. **No reflection memory** — Without carrying lessons forward, the refiner might repeat the same mistakes. Iteration 2's critique needs to know what went wrong in iteration 1.

5. **Same temperature for all roles** — The repo uses different temperatures: Generator=0.4 (creative), Critic=0.0 (deterministic), Refiner=0.3 (focused). Using high temperature for the critic would give inconsistent scores.

6. **Infinite loops** — Always set `max_iterations`. The LLM might never reach the pass threshold, so you need a hard stop. The repo defaults to 3.

7. **Putting the rubric in the system prompt** — The rubric and constitution go in the user message where there's more room and the LLM pays more attention to it. The system prompt stays short and role-defining.

---

## Practice Exercises

1. **Change the pass score to 6** — Edit `pass_score=6` in `main.py` and run. Does the weak draft pass on the first iteration? What does this tell you about threshold sensitivity?

2. **Write a new constitution** — Create a `data/constitution_strict.md` with rules like "Never use exclamation marks" or "Every claim must cite a specific program feature." Run the loop and see how the output changes.

3. **Add a new rubric criterion** — Edit `data/rubric.md` to add "6. Emotional tone — should be warm but not pushy." Run the loop and check if the critic catches tone issues.

4. **Remove reflection memory** — In `critic_loop.py`, change `reflections.append(...)` to `pass` (so reflections stay empty). Run the loop and compare output quality — does the refiner make the same mistakes repeatedly?

5. **Build your own self-reflection loop** — Write a new `main.py` that evaluates code documentation instead of marketing copy. Create a new rubric with criteria like "completeness," "clarity," and "example quality." Use the same schemas and loop structure.
