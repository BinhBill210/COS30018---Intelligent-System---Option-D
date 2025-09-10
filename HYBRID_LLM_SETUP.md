# Hybrid LLM Architecture Setup Guide

This guide will help you set up and use the hybrid LLM architecture that integrates Google's Gemini API with your existing local LLM for business intelligence.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements_hybrid.txt
```

### 2. Configure API Keys (Optional for Gemini)

**Option A: Using Environment Variables**
```bash
export GEMINI_API_KEY="your_gemini_api_key_here"
```

**Option B: Using the Interactive Setup**
```bash
python config/api_keys.py
```

**Option C: Using the Streamlit Interface**
- Run the application and use the API Key Management section in the sidebar

### 3. Run the Application

```bash
streamlit run streamlit_agent.py
```

## 🏗️ Architecture Overview

### Components

1. **Gemini LLM Wrapper** (`gemini_llm.py`)
   - Implements LangChain LLM interface
   - Handles API authentication and configuration
   - Provides connection testing capabilities

2. **Hybrid LLM System** (`langchain_agent_chromadb.py`)
   - Supports local, Gemini, and hybrid modes
   - Automatic fallback between models
   - Performance tracking and logging

3. **API Key Management** (`config/api_keys.py`)
   - Secure storage using system keyring or encrypted files
   - Key validation and rotation support
   - Environment variable fallback

4. **Logging & Monitoring** (`config/logging_config.py`)
   - Comprehensive performance tracking
   - Model switch logging
   - Error monitoring and analysis

5. **Enhanced Streamlit Interface** (`streamlit_agent.py`)
   - Model selection UI
   - Real-time performance stats
   - API key management
   - Error handling and status display

## 🔧 Configuration Options

### Model Types

1. **Local Only**
   - Uses Qwen2.5 model running locally
   - Fast, private, no API costs
   - Works offline

2. **Gemini Only**
   - Uses Google's Gemini API
   - More powerful reasoning
   - Requires API key and internet

3. **Hybrid Mode**
   - Primary model with fallback option
   - Best of both worlds
   - Automatic error recovery

### Gemini Configuration

```python
from gemini_llm import GeminiConfig

config = GeminiConfig(
    model_name="gemini-1.5-flash",  # or "gemini-1.5-pro"
    temperature=0.1,
    max_output_tokens=2048,
    top_p=0.95,
    top_k=40
)
```

## 🎯 Usage Examples

### Basic Usage (Local Model)

```python
from langchain_agent_chromadb import create_business_agent_chromadb

# Create agent with local model
agent = create_business_agent_chromadb(model_type="local")

# Run query
response = agent.invoke({
    "input": "What are customers saying about food quality?",
    "chat_history": ""
})
```

### Gemini API Usage

```python
# Create agent with Gemini
agent = create_business_agent_chromadb(
    model_type="gemini",
    gemini_config=GeminiConfig(temperature=0.1)
)
```

### Hybrid Usage with Fallback

```python
# Create hybrid agent (Gemini primary, Local fallback)
agent = create_business_agent_chromadb(
    model_type="hybrid",
    primary_model="gemini",
    fallback_model="local"
)
```

## 📊 Performance Monitoring

### Real-time Stats in Streamlit
- Success rates by model
- Average response times
- Fallback usage frequency
- Error tracking

### Log Analysis

```python
from config.logging_config import get_performance_logger

perf_logger = get_performance_logger()
stats = perf_logger.get_performance_summary(hours=24)
print(stats)
```

### Log Files
- `logs/hybrid_llm.log` - General application logs
- `logs/llm_performance.jsonl` - Detailed performance metrics
- `logs/model_switches.jsonl` - Model switch events
- `logs/errors.log` - Error logs

## 🔒 Security Features

### API Key Management
- Encrypted storage of API keys
- System keyring integration
- Secure permission settings
- Key validation and testing

### Fallback Safety
- Graceful error handling
- No data loss on model failures
- Automatic model switching
- Error message sanitization

## 🛠️ Troubleshooting

### Common Issues

1. **Gemini API Key Issues**
   ```
   Error: Gemini API key not found
   ```
   **Solution**: Configure your API key using one of the methods above

2. **Local Model Loading Issues**
   ```
   Error: Could not load local model
   ```
   **Solution**: Ensure you have sufficient RAM and GPU memory (if using CUDA)

3. **ChromaDB Issues**
   ```
   Error: ChromaDB connection failed
   ```
   **Solution**: Check that your ChromaDB files exist and are accessible

### Performance Optimization

1. **For Local Models**
   - Use 4-bit quantization if memory is limited
   - Ensure CUDA is available for GPU acceleration
   - Consider using smaller models for faster inference

2. **For Gemini API**
   - Monitor API usage and costs
   - Use appropriate temperature settings
   - Consider caching frequent queries

3. **For Hybrid Mode**
   - Set appropriate timeout values
   - Monitor fallback usage patterns
   - Balance cost vs. performance

## 📈 Advanced Features

### Custom Model Configuration

```python
from gemini_llm import GeminiConfig

custom_config = GeminiConfig(
    model_name="gemini-1.5-pro",
    temperature=0.2,
    max_output_tokens=4096,
    top_p=0.9,
    top_k=50
)

agent = create_business_agent_chromadb(
    model_type="hybrid",
    primary_model="gemini",
    fallback_model="local",
    gemini_config=custom_config
)
```

### Performance Analytics

```python
# Get detailed performance analysis
from config.logging_config import get_performance_logger

logger = get_performance_logger()
summary = logger.get_performance_summary(hours=24)

for model, stats in summary.items():
    if model.startswith('_'):
        continue
    print(f"{model}:")
    print(f"  Success Rate: {stats['success_rate']:.1%}")
    print(f"  Avg Response Time: {stats['avg_generation_time']:.2f}s")
    print(f"  Tokens/Second: {stats['avg_tokens_per_second']:.1f}")
```

## 🔄 Migration from Existing Setup

### From Pure Local Setup
1. Install new dependencies
2. Update imports in your existing code
3. Configure model selection parameters
4. Test with hybrid mode for gradual transition

### From Pure API Setup
1. Install local model dependencies
2. Configure local model as fallback
3. Set up hybrid mode with API as primary
4. Monitor cost savings with fallback usage

## 📝 Best Practices

1. **Model Selection**
   - Use local for simple queries and privacy
   - Use Gemini for complex reasoning tasks
   - Use hybrid for balanced performance and reliability

2. **Error Handling**
   - Always configure fallback models
   - Monitor error rates and patterns
   - Set appropriate timeout values

3. **Cost Management**
   - Monitor Gemini API usage
   - Use local model for development/testing
   - Implement usage limits if needed

4. **Performance**
   - Regular monitoring of response times
   - Optimize model configurations
   - Cache frequent queries when possible

## 🆘 Support

For issues or questions:
1. Check the logs in the `logs/` directory
2. Review the Streamlit debug panel
3. Test API connections using the built-in tools
4. Monitor performance statistics for patterns

## 🔮 Future Enhancements

- Support for additional LLM providers (OpenAI, Anthropic, etc.)
- Advanced caching mechanisms
- Load balancing across multiple instances
- Cost optimization algorithms
- Enhanced monitoring dashboards
