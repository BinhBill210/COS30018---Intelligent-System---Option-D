# LangSmith Evaluation Flow - Visual Guide

## 🔄 Evaluation Process Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    EVALUATION PIPELINE                       │
└─────────────────────────────────────────────────────────────┘

   ┌─────────────────┐
   │  1. Load Test   │
   │     Dataset     │
   │  (JSON file)    │
   └────────┬────────┘
            │
            ▼
   ┌─────────────────┐
   │  2. Create or   │
   │  Load LangSmith │
   │     Dataset     │
   └────────┬────────┘
            │
            ▼
   ┌─────────────────┐
   │  3. Initialize  │
   │   G1 Agent      │
   │  (@traceable)   │
   └────────┬────────┘
            │
            ▼
   ┌─────────────────────────────────────────┐
   │  4. FOR EACH TEST CASE:                 │
   │                                         │
   │  ┌─────────────────────────────────┐   │
   │  │  a. Run Agent on Query          │   │
   │  └────────────┬────────────────────┘   │
   │               │                         │
   │               ▼                         │
   │  ┌─────────────────────────────────┐   │
   │  │  b. Track Tool Calls            │   │
   │  │     (via @traceable)            │   │
   │  └────────────┬────────────────────┘   │
   │               │                         │
   │               ▼                         │
   │  ┌─────────────────────────────────┐   │
   │  │  c. Capture Output              │   │
   │  └────────────┬────────────────────┘   │
   │               │                         │
   │               ▼                         │
   │  ┌─────────────────────────────────┐   │
   │  │  d. Run Evaluators              │   │
   │  │     - Tool Sequence Match       │   │
   │  │     - Answer Quality            │   │
   │  └────────────┬────────────────────┘   │
   │               │                         │
   │               ▼                         │
   │  ┌─────────────────────────────────┐   │
   │  │  e. Store Results in LangSmith  │   │
   │  └─────────────────────────────────┘   │
   └─────────────────────────────────────────┘
            │
            ▼
   ┌─────────────────┐
   │  5. Aggregate   │
   │    Results      │
   └────────┬────────┘
            │
            ▼
   ┌─────────────────┐
   │  6. Display     │
   │    Summary      │
   └─────────────────┘
```

## 📊 Data Flow for a Single Test Case

```
INPUT (Test Case)
─────────────────
{
  "query": "Find the business_id for 'Body Cycle Spinning Studio'",
  "expected_tool_chain": ["business_fuzzy_search"],
  "expected_answer_summary": ["Returns the business ID..."]
}

            │
            ▼

AGENT EXECUTION (@traceable)
────────────────────────────
run_g1_agent(query) {
  1. Agent receives query
  2. Agent plans action
  3. Agent calls tools ──┐
     └─> business_fuzzy_search(@traceable)
         └─> returns business_id
  4. Agent generates final answer
}

            │
            ▼

OUTPUT (Agent Response)
───────────────────────
{
  "answer": "The business ID is 7ATYjTfgJjUIt4UN3IypQ",
  "tool_calls": ["business_fuzzy_search"],
  "query": "..."
}

            │
            ▼

EVALUATION
──────────
┌─ Evaluator 1: Tool Sequence Match ─────────────────┐
│                                                     │
│  Expected: ["business_fuzzy_search"]                │
│  Actual:   ["business_fuzzy_search"]                │
│                                                     │
│  ✓ Exact match!                                     │
│  Score: 1.0                                         │
└─────────────────────────────────────────────────────┘

┌─ Evaluator 2: Answer Quality ──────────────────────┐
│                                                     │
│  Answer length: 45 chars                            │
│  Contains "business ID": ✓                          │
│  Not an error: ✓                                    │
│                                                     │
│  ✓ Valid answer                                     │
│  Score: 1.0                                         │
└─────────────────────────────────────────────────────┘

            │
            ▼

RESULT STORED IN LANGSMITH
──────────────────────────
✓ Test case passed
✓ All metrics: 1.0
✓ Trace available for inspection
```

## 🔍 What @traceable Does

```python
@traceable(name="My_Function")
def my_function(input: str) -> str:
    # LangSmith automatically captures:
    # 1. Input parameters
    # 2. Output value
    # 3. Execution time
    # 4. Any errors
    # 5. Child function calls (if they're also @traceable)
    
    result = some_processing(input)
    return result
```

**Visual Trace:**
```
┌─ my_function (traced) ───────────────────────────┐
│                                                   │
│  Input: "user query"                              │
│  Start: 10:30:00.000                              │
│                                                   │
│  ┌─ child_function (traced) ──────────────────┐  │
│  │  Input: "processed query"                   │  │
│  │  Output: "result"                            │  │
│  │  Duration: 0.5s                              │  │
│  └──────────────────────────────────────────────┘  │
│                                                   │
│  Output: "final result"                           │
│  End: 10:30:01.200                                │
│  Duration: 1.2s                                   │
│  Status: ✓ Success                                │
└───────────────────────────────────────────────────┘
```

## 📈 Evaluation Metrics Explained

### Metric 1: Tool Sequence Match

**Purpose**: Measures if the agent uses the right tools in the right order.

**Scoring Logic**:
```python
Expected: ["tool_A", "tool_B", "tool_C"]
Actual:   ["tool_A", "tool_B", "tool_C"]
➜ Score: 1.0 (Perfect match)

Expected: ["tool_A", "tool_B"]
Actual:   ["tool_B", "tool_A"]
➜ Score: 0.5 (All tools present, wrong order)

Expected: ["tool_A", "tool_B"]
Actual:   ["tool_A"]
➜ Score: 0.0 (Missing tools)
```

**Why it matters**: Shows if your agent has the right "capabilities" - can it identify and execute the correct workflow?

### Metric 2: Answer Quality

**Purpose**: Measures if the agent produces a valid, non-empty answer.

**Scoring Logic**:
```python
Answer: "The business ID is 7ATYjTfgJjUIt4UN3IypQ"
➜ Score: 1.0 (Valid answer)

Answer: ""
➜ Score: 0.0 (Empty answer)

Answer: "Error: Business not found"
➜ Score: 0.0 (Error response)
```

**Why it matters**: Shows if your agent has the right "behavior" - does it produce useful outputs?

## 🎯 Example Evaluation Run

```
Test Dataset: 100 test cases
─────────────────────────────

┌─ Test Case #1 ─────────────────────────────────────┐
│ Query: "Find business_id for 'Body Cycle...'"      │
│ Tool Sequence Match: 1.0 ✓                         │
│ Answer Quality: 1.0 ✓                              │
└─────────────────────────────────────────────────────┘

┌─ Test Case #2 ─────────────────────────────────────┐
│ Query: "Find business name for 'od6skmf...'"       │
│ Tool Sequence Match: 1.0 ✓                         │
│ Answer Quality: 1.0 ✓                              │
└─────────────────────────────────────────────────────┘

┌─ Test Case #3 ─────────────────────────────────────┐
│ Query: "Give me a summary about..."                │
│ Tool Sequence Match: 0.5 ⚠                         │
│   ↳ Used right tools but wrong order               │
│ Answer Quality: 1.0 ✓                              │
└─────────────────────────────────────────────────────┘

...

┌─ Aggregate Results ────────────────────────────────┐
│                                                     │
│  Total Test Cases: 100                              │
│                                                     │
│  Average Tool Sequence Match: 0.87                  │
│  Average Answer Quality: 0.95                       │
│                                                     │
│  Perfect Matches: 85/100 (85%)                      │
│  Partial Matches: 12/100 (12%)                      │
│  Failures: 3/100 (3%)                               │
│                                                     │
└─────────────────────────────────────────────────────┘
```

## 🔗 Where Everything Connects

```
┌─────────────────────────────────────────────────────────┐
│                   YOUR LOCAL MACHINE                     │
│                                                          │
│  ┌───────────────────────────────────────────────┐     │
│  │  simple_langsmith_eval_with_agent.py          │     │
│  │                                               │     │
│  │  • Loads test dataset                         │     │
│  │  • Initializes G1 agent                       │     │
│  │  • Runs evaluation                            │     │
│  │  • Sends traces to LangSmith ────────────┐    │     │
│  └───────────────────────────────────────────────┘    │
│                                                  │     │
└──────────────────────────────────────────────────│─────┘
                                                   │
                                                   │ (HTTPS)
                                                   │
                                                   ▼
┌─────────────────────────────────────────────────────────┐
│                  LANGSMITH CLOUD                         │
│                  (smith.langchain.com)                   │
│                                                          │
│  ┌─────────────────┐  ┌──────────────────┐             │
│  │   Datasets      │  │   Experiments    │             │
│  │                 │  │                  │             │
│  │  • Test cases   │  │  • Eval runs     │             │
│  │  • Examples     │  │  • Scores        │             │
│  └─────────────────┘  └──────────────────┘             │
│                                                          │
│  ┌─────────────────┐  ┌──────────────────┐             │
│  │    Traces       │  │   Dashboards     │             │
│  │                 │  │                  │             │
│  │  • Tool calls   │  │  • Visualize     │             │
│  │  • Timing       │  │  • Compare       │             │
│  └─────────────────┘  └──────────────────┘             │
│                                                          │
└─────────────────────────────────────────────────────────┘
                            ▲
                            │
                            │ (View in Browser)
                            │
                      ┌─────────────┐
                      │    YOU      │
                      │             │
                      │ • Review    │
                      │ • Analyze   │
                      │ • Export    │
                      └─────────────┘
```

## 💡 Key Takeaways

1. **@traceable is crucial**: Without it, LangSmith can't track your agent's behavior
2. **Evaluators are flexible**: You can create custom evaluators for any aspect you want to measure
3. **Results are stored**: All evaluation data is saved in LangSmith for future reference
4. **Traces are visual**: You can see exactly what your agent did, step by step
5. **Iterate and improve**: Use evaluation results to identify weaknesses and improve your agent

## 🚀 Next Steps After Running Evaluation

1. **Review dashboard**: Look at individual test cases that failed
2. **Examine traces**: See exactly what tools were called and why
3. **Identify patterns**: Are certain categories failing more than others?
4. **Improve agent**: Fix the issues you found
5. **Re-evaluate**: Run the evaluation again to measure improvement
6. **Compare runs**: Use LangSmith to compare before/after results

---

**Ready to start?** Go to `README_LANGSMITH.md` for step-by-step instructions!

