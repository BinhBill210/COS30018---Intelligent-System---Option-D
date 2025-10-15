# LangSmith Evaluation for G1 Agent

This directory contains simple, student-friendly scripts to evaluate your G1 LangChain agent using LangSmith.

## 📁 Files Overview

| File | Description |
|------|-------------|
| `simple_langsmith_eval.py` | **⭐ MAIN SCRIPT** - Evaluates your actual G1 agent with LangSmith |
| `simple_langsmith_eval_with_agent.py` | **Alternative version** - Same functionality as above |
| `test_langsmith_setup.py` | **Setup checker** - Verifies your environment is configured correctly |
| `run_langsmith_eval.bat` | **Windows launcher** - Batch script to run evaluation (Command Prompt) |
| `run_langsmith_eval.ps1` | **Windows launcher** - PowerShell script to run evaluation |
| `LANGSMITH_EVAL_GUIDE.md` | **Detailed guide** - In-depth explanation of the evaluation system |
| `CHANGES_ACTUAL_AGENT.md` | **⭐ NEW** - Explains the actual agent integration |

## 🚀 Quick Start (3 Steps)

### Step 1: Get Your LangSmith API Key

1. Go to https://smith.langchain.com/
2. Sign up or log in
3. Go to Settings → API Keys
4. Create a new API key
5. Copy the key

### Step 2: Set Environment Variables

**Windows PowerShell:**
```powershell
# Set LangSmith API key (required)
$env:LANGSMITH_API_KEY = "your-langsmith-api-key-here"

# Optional: Set Gemini API key if using Gemini LLM
$env:GEMINI_API_KEY = "your-gemini-api-key-here"

# Enable LangSmith tracing
$env:LANGCHAIN_TRACING_V2 = "true"
```

**Windows Command Prompt:**
```batch
REM Set LangSmith API key (required)
set LANGSMITH_API_KEY=your-langsmith-api-key-here

REM Optional: Set Gemini API key
set GEMINI_API_KEY=your-gemini-api-key-here

REM Enable tracing
set LANGCHAIN_TRACING_V2=true
```

**Linux/Mac:**
```bash
# Set LangSmith API key (required)
export LANGSMITH_API_KEY="your-langsmith-api-key-here"

# Optional: Set Gemini API key
export GEMINI_API_KEY="your-gemini-api-key-here"

# Enable tracing
export LANGCHAIN_TRACING_V2=true
```

### Step 3: Run the Evaluation

#### Option A: Use the Launcher Scripts (Easiest)

**Windows - PowerShell:**
```powershell
cd E:\COS30018---Intelligent-System---Option-D
.\evaluation\run_langsmith_eval.ps1
```

**Windows - Command Prompt:**
```batch
cd E:\COS30018---Intelligent-System---Option-D
evaluation\run_langsmith_eval.bat
```

The launcher will:
1. Check your environment variables
2. Activate the conda environment (`langchain-demo`)
3. Let you choose which evaluation to run

#### Option B: Run Scripts Manually

**First Time - Test Your Setup:**
```powershell
conda activate langchain-demo
cd E:\COS30018---Intelligent-System---Option-D
python evaluation/test_langsmith_setup.py
```

**Run G1 Agent Evaluation (Main Script):**
```powershell
conda activate langchain-demo

# Make sure services are running if needed
# python scripts/start_chroma_servers.py

# Run the evaluation with your actual G1 agent
python evaluation/simple_langsmith_eval.py
```

**Note:** Both `simple_langsmith_eval.py` and `simple_langsmith_eval_with_agent.py` now use the actual G1 agent implementation from `langchain_agent_chromadb.py`.

## 📊 Understanding the Evaluation

### What Gets Evaluated?

The evaluation tests your G1 agent on **test cases** from `golden_test_dataset_v2.json`.

Each test case has:
- **Input**: A query (e.g., "Find the business_id for 'Body Cycle Spinning Studio'")
- **Expected tool chain**: Which tools should be called (e.g., `["business_fuzzy_search"]`)
- **Expected answer**: What the answer should contain

### Evaluation Metrics

The scripts evaluate two key aspects:

#### 1. **Agent Capabilities** - Tool Sequence Match
- **What it measures**: Does the agent use the right tools in the right order?
- **Scoring**:
  - 1.0 = Perfect match (exact sequence)
  - 0.5 = Partial match (all tools used, wrong order)
  - 0.0 = Missing or wrong tools

#### 2. **Agent Behavior** - Answer Quality
- **What it measures**: Does the agent produce a valid answer?
- **Scoring**:
  - 1.0 = Valid answer provided
  - 0.0 = No answer or error

### Where to View Results

After running the evaluation:

1. **Console Output**: See summary of results
2. **LangSmith Dashboard**: 
   - Go to https://smith.langchain.com/
   - Navigate to project: "G1 Agent Evaluation"
   - View:
     - Individual test runs
     - Tool call traces (execution flow)
     - Evaluator scores
     - Aggregate metrics
     - Errors and debugging info

## 🔧 Customization

### Modify the Placeholder Agent

Edit `simple_langsmith_eval.py`, function `run_g1_agent()`:

```python
@traceable(name="G1_Agent", project_name=PROJECT_NAME)
def run_g1_agent(query: str) -> Dict[str, Any]:
    # Replace this with your agent logic
    # Make sure to return:
    # - answer: str (the final answer)
    # - tool_calls: List[str] (tools that were used)
    
    return {
        "answer": "Your answer here",
        "tool_calls": ["tool1", "tool2"]
    }
```

### Add Custom Evaluators

Add new evaluator functions in either script:

```python
def my_custom_evaluator(run: Run, example: Example) -> dict:
    """
    Custom evaluator for [describe what it checks].
    """
    # Your evaluation logic here
    
    return {
        "key": "my_metric_name",
        "score": 0.0,  # 0.0 to 1.0
        "comment": "Explanation of the score"
    }
```

Then add it to the `evaluators` list in `run_evaluation()`:

```python
results = client.evaluate(
    ...,
    evaluators=[
        exact_tool_sequence_evaluator,
        answer_quality_evaluator,
        my_custom_evaluator  # Add your custom evaluator here
    ],
    ...
)
```

### Change LLM Type

In `simple_langsmith_eval_with_agent.py`, modify the `load_g1_agent()` function:

```python
def load_g1_agent():
    from langchain_agent_chromadb import create_langchain_agent
    
    # Change "gemini" to "local" to use local LLM
    agent = create_langchain_agent(llm_type="gemini")  # or "local"
    
    return agent
```

## 📝 Important Notes

### The @traceable Decorator

This decorator is **critical** for LangSmith to track your agent's behavior:

```python
from langsmith.run_helpers import traceable

@traceable(name="My_Function")
def my_function(input: str) -> str:
    # LangSmith will automatically track:
    # - Inputs and outputs
    # - Execution time
    # - Errors
    # - Child function calls (if they're also @traceable)
    return process(input)
```

**Where to use it:**
- ✅ Your main agent function
- ✅ Individual tool functions
- ✅ Helper functions you want to track
- ❌ Don't overuse on every small utility function

### Dataset Management

**First run**: Creates a new dataset in LangSmith

**Subsequent runs**: Uses the existing dataset

**To recreate dataset**:
- Delete it in LangSmith dashboard, OR
- Change `DATASET_NAME` in the script

### Cost Considerations

- **Placeholder agent**: Free (no API calls)
- **Local LLM agent**: Free (uses local GPU/CPU)
- **Gemini agent**: Uses Gemini API (check your quota)

Each test case runs once per evaluation.

## 🐛 Troubleshooting

### "LANGSMITH_API_KEY not set"

Set the environment variable:
```powershell
$env:LANGSMITH_API_KEY = "your-key-here"
```

### "Dataset already exists"

This is normal. The script will use the existing dataset. To force recreation:
1. Delete dataset in LangSmith dashboard, OR
2. Change `DATASET_NAME` in the script

### "Failed to load agent"

Check that:
- ✅ ChromaDB is running (if using ChromaDB)
- ✅ All required API keys are set
- ✅ Conda environment is activated
- ✅ Dependencies are installed (`pip install langsmith`)

### "No module named 'langsmith'"

Install LangSmith:
```bash
conda activate langchain-demo
pip install langsmith
```

### Evaluation runs but all scores are 0

This is expected with the **placeholder agent**. The placeholder returns mock data that doesn't match the expected outputs.

To get real scores:
1. Use `simple_langsmith_eval_with_agent.py` (with real agent), OR
2. Modify the placeholder to return realistic data for testing

## 📚 Additional Resources

- **LangSmith Docs**: https://docs.smith.langchain.com/
- **Evaluation Guide**: https://docs.smith.langchain.com/evaluation
- **Tracing Guide**: https://docs.smith.langchain.com/tracing
- **Detailed Guide**: See `LANGSMITH_EVAL_GUIDE.md` in this directory

## 🎯 Recommended Workflow

1. **First time**: Run `test_langsmith_setup.py` to verify setup
2. **Test framework**: Run `simple_langsmith_eval.py` (placeholder) to test the evaluation process
3. **Real evaluation**: Run `simple_langsmith_eval_with_agent.py` with your actual agent
4. **Review results**: Check LangSmith dashboard
5. **Improve agent**: Fix issues found in evaluation
6. **Re-evaluate**: Run evaluation again to measure improvements

## 💡 Tips for Students

- Start with the placeholder version to understand the flow
- Read the comments in the code - they explain each step
- Use LangSmith dashboard to visualize your agent's behavior
- Try different evaluators to measure different aspects
- Keep evaluation runs for comparison (before/after improvements)
- Export results for your project report

---

**Questions?** Check `LANGSMITH_EVAL_GUIDE.md` for more detailed explanations.

