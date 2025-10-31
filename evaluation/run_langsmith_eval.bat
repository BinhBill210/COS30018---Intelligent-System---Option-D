@echo off
REM Batch Script to Run LangSmith Evaluation

echo ============================================================
echo LangSmith Evaluation Runner
echo ============================================================

REM Check if API key is set
if not defined LANGSMITH_API_KEY (
    echo.
    echo ERROR: LANGSMITH_API_KEY is not set!
    echo Please set it first:
    echo   set LANGSMITH_API_KEY=your-api-key-here
    echo.
    pause
    exit /b 1
)

echo.
echo [Step 1] LANGSMITH_API_KEY is set
echo [Step 2] Activating conda environment...

REM Activate conda environment
call conda activate langchain-demo
if errorlevel 1 (
    echo ERROR: Failed to activate conda environment
    echo Make sure 'langchain-demo' or 'biz-agent-gpu-2' environment exists
    pause
    exit /b 1
)

echo.
echo [Step 3] Choose evaluation mode:
echo   1. Test setup (recommended first time^)
echo   2. Run G1 agent evaluation (with actual agent^)
echo.

set /p choice="Enter choice (1 or 2): "

if "%choice%"=="1" (
    echo.
    echo Running setup test...
    python evaluation\test_langsmith_setup.py
) else if "%choice%"=="2" (
    echo.
    echo Running G1 agent evaluation...
    echo NOTE: Make sure ChromaDB and other services are running!
    echo Using actual G1 agent from langchain_agent_chromadb.py
    python evaluation\simple_langsmith_eval.py
) else (
    echo.
    echo Invalid choice. Exiting.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo Done!
echo ============================================================
pause

