# PowerShell Script to Run LangSmith Evaluation
# ===============================================

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "LangSmith Evaluation Runner" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan

# Check if API key is set
Write-Host "`n[Step 1] Checking environment variables..." -ForegroundColor Yellow

if (-not $env:LANGSMITH_API_KEY) {
    Write-Host "  ERROR: LANGSMITH_API_KEY is not set!" -ForegroundColor Red
    Write-Host "  Please set it first:" -ForegroundColor Yellow
    Write-Host "    `$env:LANGSMITH_API_KEY = 'your-api-key-here'" -ForegroundColor White
    exit 1
}

Write-Host "  OK: LANGSMITH_API_KEY is set" -ForegroundColor Green

# Activate conda environment
Write-Host "`n[Step 2] Activating conda environment..." -ForegroundColor Yellow
Write-Host "  Activating 'langchain-demo' or 'biz-agent-gpu-2' environment..." -ForegroundColor Cyan

# Choose which script to run
Write-Host "`n[Step 3] Choose evaluation mode:" -ForegroundColor Yellow
Write-Host "  1. Test setup (recommended first time)" -ForegroundColor White
Write-Host "  2. Run G1 agent evaluation (with actual agent)" -ForegroundColor White

$choice = Read-Host "`nEnter choice (1 or 2)"

switch ($choice) {
    "1" {
        Write-Host "`nRunning setup test..." -ForegroundColor Cyan
        python evaluation\test_langsmith_setup.py
    }
    "2" {
        Write-Host "`nRunning G1 agent evaluation..." -ForegroundColor Cyan
        Write-Host "  NOTE: Make sure ChromaDB and other services are running!" -ForegroundColor Yellow
        Write-Host "  Using actual G1 agent from langchain_agent_chromadb.py" -ForegroundColor Cyan
        python evaluation\simple_langsmith_eval.py
    }
    default {
        Write-Host "`nInvalid choice. Exiting." -ForegroundColor Red
        exit 1
    }
}

Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host "Done!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan

