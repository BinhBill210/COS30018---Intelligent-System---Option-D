# LangChain Demo Setup Guide

This guide helps you set up the environment to run the LangChain business review analysis demo.

## Quick Start

### Option 1: GPU Environment (Recommended if you have CUDA GPU)

```bash
# Create the environment
conda env create -f langchain-demo-env.yml

# Activate the environment
conda activate langchain-demo

# Run the demo
python demo2_langchain.py
```

### Option 2: CPU-Only Environment

```bash
# Create the CPU-only environment
conda env create -f langchain-demo-cpu.yml

# Activate the environment
conda activate langchain-demo-cpu

# Run the demo
python demo2_langchain.py
```

## Prerequisites

1. **Conda or Miniconda**: Make sure you have conda installed
   - Download from: https://docs.conda.io/en/latest/miniconda.html

2. **Data Files**: Ensure you have the required data files:
   - `data/processed/review_cleaned.csv` - The review dataset
   - `index_demo/` directory - The FAISS search index (optional)

3. **GPU Requirements** (for GPU version):
   - NVIDIA GPU with CUDA 11.8 support
   - CUDA drivers installed

## Environment Details

### GPU Environment (`langchain-demo-env.yml`)
- **Python**: 3.10
- **PyTorch**: 2.1.2+cu118 (CUDA 11.8)
- **FAISS**: GPU-accelerated version
- **LangChain**: 0.3.27+
- **Transformers**: 4.55.0+

### CPU Environment (`langchain-demo-cpu.yml`)
- **Python**: 3.10
- **PyTorch**: CPU-only version
- **FAISS**: CPU version
- **LangChain**: 0.3.27+
- **Transformers**: 4.55.0+

## Troubleshooting

### Common Issues

1. **"RuntimeError: operator torchvision::nms does not exist"**
   - Solution: Use the CPU environment instead
   ```bash
   conda env create -f langchain-demo-cpu.yml
   ```

2. **"ImportError: cannot import name 'cached_download'"**
   - Solution: This is fixed in the new environment files

3. **Model download issues**
   - Some models may require HuggingFace authentication
   - Get a token from: https://huggingface.co/settings/tokens
   - Set it as environment variable:
   ```bash
   export HF_TOKEN=your_token_here
   ```

4. **Out of memory errors**
   - Use smaller models or CPU-only environment
   - The demo includes fallbacks for when models fail to load

### Verification

Test that everything works:

```bash
# Activate your environment
conda activate langchain-demo  # or langchain-demo-cpu

# Test basic imports
python -c "import langchain; from langchain_agent import create_business_agent; print('✅ Setup successful!')"

# Run the full demo
python demo2_langchain.py

# NEW: Run ChromaDB demo (with vector search)
python demo_chromadb.py
```

## ChromaDB Migration (NEW!)

This project has been **upgraded** from custom FAISS to **ChromaDB + LangChain** integration:

### Migration Steps
```bash
# 1. Migrate existing data to ChromaDB (one-time setup)
python migrate_to_chromadb.py

# 2. Test the new ChromaDB system
python demo_chromadb.py

# 3. Interactive mode with vector search
python enhanced_chromadb_agent.py interactive
```

### Benefits of ChromaDB
- 🔍 **Semantic Search**: Find relevant reviews by meaning, not just keywords
- 💾 **Persistent Storage**: No need to rebuild index every time
- 🔧 **Better Integration**: Native LangChain compatibility
- 📊 **Metadata Filtering**: Filter by business, date, rating, etc.
- 🚀 **Better Performance**: More efficient than custom FAISS implementation

## What the Demo Does

The LangChain demo now includes **ChromaDB vector search** capabilities:

### Legacy Demo (`demo2_langchain.py`)
1. **Sentiment Analysis**: Analyzes the sentiment of business reviews
2. **Data Summarization**: Provides statistical summaries of review data
3. **LangChain Integration**: Shows how to use LangChain agents with custom tools

### New ChromaDB Demo (`demo_chromadb.py` & `enhanced_chromadb_agent.py`)
1. **🔍 Vector Search**: Find relevant reviews using semantic similarity
2. **💬 RAG Question-Answering**: Answer complex questions using retrieved reviews
3. **😊 Sentiment Analysis**: Analyze sentiment patterns in review texts
4. **📊 Data Summaries**: Get statistical insights with filtering capabilities

Sample queries for ChromaDB version:
- "What do customers say about food quality?"
- "Find reviews mentioning poor service and analyze sentiment"
- "What are the main complaints customers have?"
- "Give me statistics for business XQfwVwDr-v0ZS3_CbbE5Xw"

## Customization

### Using Different Models

You can modify `langchain_agent.py` to use different models:

```python
# For smaller, faster models
model_name = "microsoft/DialoGPT-small"

# For better quality (requires more memory)
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
```

### Adding New Tools

Add new tools to `langchain_agent.py` following the LangChain Tool pattern:

```python
class MyCustomTool(BaseTool):
    name: str = "my_tool"
    description: str = "What my tool does"
    
    def _run(self, input_param: str) -> str:
        # Your tool logic here
        return "Tool result"
```

## Performance Tips

1. **GPU vs CPU**: GPU environment is ~5-10x faster for model inference
2. **Model Size**: Smaller models load faster but may have lower quality
3. **Memory**: Close other applications when running large models
4. **Batch Processing**: Process multiple reviews at once for better efficiency

## Support

If you encounter issues:

1. Check the troubleshooting section above
2. Verify your environment matches the working setup
3. Try the CPU environment if GPU version fails
4. Check that all required data files are present

For additional help, refer to:
- [LangChain Documentation](https://python.langchain.com/)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/)
- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
