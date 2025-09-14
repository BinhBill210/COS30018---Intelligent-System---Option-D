# 🚀 Hybrid LLM Business Intelligence System

A comprehensive business intelligence system that combines local and cloud-based language models for analyzing business reviews and generating insights. This system integrates **local Qwen2.5 models** with **Google's Gemini API** using LangChain, ChromaDB for RAG capabilities, and a beautiful Streamlit interface.

## 🌟 Key Features

- **🔄 Hybrid Model Architecture**: Choose between local models, Gemini API, or intelligent hybrid mode with automatic fallback
- **🏢 Business Intelligence Tools**: Advanced sentiment analysis, aspect-based analysis, business search, and data summarization
- **💾 RAG-Powered**: ChromaDB vector database for semantic search across business reviews
- **🔐 Secure API Management**: Encrypted storage and management of API keys
- **📊 Performance Monitoring**: Real-time tracking of model performance and usage statistics
- **🎨 Modern UI**: Beautiful Streamlit interface with model selection and status monitoring

## 📋 System Requirements

- **Python**: 3.10+
- **Memory**: 8GB+ RAM (16GB+ recommended for local models)
- **GPU**: NVIDIA GPU with CUDA (optional, for local model acceleration)
- **Storage**: 5GB+ free space

## 🛠️ Quick Start

### 1. Environment Setup

**Option A: Complete Environment (Recommended)**
```bash
# Clone the repository
git clone <your-repo-url>
cd COS30018---Intelligent-System---Option-D

# Create conda environment with all dependencies
conda env create -f environment-hybrid.yml
conda activate hybrid-llm-bi
```

**Option B: Existing Environment**
```bash
# If you have an existing environment
pip install -r requirements_hybrid.txt
```

### 2. Data Setup

**Option A: Download Pre-processed Data (Fast)**
- Download the `data` and `chroma_db` folders from [Google Drive](https://drive.google.com/drive/u/0/folders/1enrB0_dKmCJG62NjTBqRG_pZF76Xv4z9)
- Place them in the project root directory

**Option B: Process Data Yourself**
```bash
# Run data preprocessing
python scripts/run_preprocessing.py

# Create vector embeddings
python migrate_to_chromadb.py
```

### 3. API Key Setup (Optional - for Gemini)

**Option A: Environment Variable**
```bash
export GEMINI_API_KEY="your_api_key_here"
```

**Option B: Interactive Setup**
```bash
python config/api_keys.py
```

**Option C: Via Streamlit Interface**
- Use the "API Key Management" section in the sidebar

### 4. Launch the Application

```bash
streamlit run streamlit_agent.py
```

Open your browser to `http://localhost:8501` and start exploring!

## 🎯 Usage Guide

### Model Selection

1. **Local Model** 🖥️
   - **Pros**: Fast, private, no API costs, works offline
   - **Cons**: Limited reasoning compared to cloud models
   - **Best for**: Quick queries, privacy-sensitive data, development

2. **Gemini API** ☁️
   - **Pros**: Advanced reasoning, larger context window, latest capabilities
   - **Cons**: Requires API key, internet connection, usage costs
   - **Best for**: Complex analysis, detailed reasoning, production insights

3. **Hybrid Mode** 🔄
   - **Pros**: Best of both worlds, automatic fallback, optimized performance
   - **Cons**: Requires both setups
   - **Best for**: Production environments, reliability-critical applications

### Business Intelligence Capabilities

- **📊 Sentiment Analysis**: Analyze customer sentiment patterns
- **🔍 Review Search**: Semantic search across business reviews
- **🏢 Business Discovery**: Find businesses by description or similarity
- **📈 Data Insights**: Statistical summaries and trend analysis
- **🎯 Aspect Analysis**: Detailed breakdown of review aspects (food, service, etc.)

### Example Queries

```
"What are customers saying about food quality in Italian restaurants?"
"Find businesses similar to high-end coffee shops in downtown area"
"Analyze sentiment trends for delivery services over time"
"Show me the most common complaints about fast food restaurants"
```

## 🏗️ Project Structure

```
📁 COS30018---Intelligent-System---Option-D/
├── 🎨 streamlit_agent.py              # Main Streamlit interface
├── 🤖 langchain_agent_chromadb.py     # Hybrid agent implementation
├── 🌟 gemini_llm.py                   # Google Gemini integration
├── 🏠 local_llm.py                    # Local model wrapper
├── 📁 config/                         # Configuration modules
│   ├── api_keys.py                    # Secure API key management
│   └── logging_config.py              # Logging and monitoring
├── 📁 tools/                          # Business intelligence tools
│   ├── review_search_tool.py          # Semantic review search
│   ├── sentiment_summary_tool.py      # Sentiment analysis
│   ├── business_search_tool.py        # Business discovery
│   ├── data_summary_tool.py           # Statistical summaries
│   └── aspect_analysis.py             # Aspect-based analysis
├── 📁 data/                           # Processed datasets
│   └── processed/                     # Clean business and review data
├── 📁 chroma_db/                      # Vector database
├── 📁 test/                           # Test suites
├── 📁 logs/                           # Application logs
├── 🐍 environment-hybrid.yml          # Complete conda environment
├── 📦 requirements_hybrid.txt         # Pip requirements
└── 📖 HYBRID_LLM_SETUP.md            # Detailed setup guide
```

## 🧪 Testing

```bash
# Run all tests
python -m test.run_all_tests

# Run specific tool tests
python -m test.test_review_search_tool
python -m test.test_sentiment_summary_tool
python -m test.test_business_search_tool
```

## 📊 Performance Monitoring

The system includes comprehensive monitoring:

- **Real-time Metrics**: Response times, success rates, token usage
- **Model Comparison**: Performance stats between local and cloud models
- **Error Tracking**: Detailed logging of failures and fallback usage
- **Usage Analytics**: API usage and cost tracking

View these in the Streamlit sidebar under "Performance Stats".

## 🔧 Configuration

### Model Configuration

```python
# Local model settings
local_config = {
    "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
    "use_4bit": True,  # For memory efficiency
    "temperature": 0.1
}

# Gemini configuration
gemini_config = GeminiConfig(
    model_name="gemini-2.0-flash",
    temperature=0.1,
    max_output_tokens=2048
)
```

### ChromaDB Configuration

The system uses ChromaDB for vector storage with:
- **Collection**: Business reviews with metadata
- **Embeddings**: Sentence-BERT for semantic similarity
- **Metadata**: Business info, ratings, categories, locations

## 🛡️ Security Features

- **🔐 Encrypted API Key Storage**: Keys stored securely using system keyring or encrypted files
- **🚫 No Hardcoded Secrets**: All sensitive data externalized
- **📝 Audit Logging**: Complete audit trail of all operations
- **🔄 Key Rotation**: Easy API key updates without code changes

## 📈 Advanced Features

### Custom Tool Integration

```python
# Add custom business intelligence tools
from tools.custom_tool import CustomAnalysisTool

custom_tool = CustomAnalysisTool()
# Integrate with existing agent
```

### API Extensions

- **REST API**: Programmatic access to all features
- **Batch Processing**: Process multiple queries efficiently
- **Webhook Integration**: Real-time notifications
- **Export Capabilities**: Results in multiple formats

## 🐛 Troubleshooting

### Agent Issues (NEW)

1. **Agent Stops Without Output (Iteration Limit)**
   ```bash
   # Run the troubleshooting script
   python troubleshoot_agent.py
   ```
   **Or in Streamlit**: Increase "Max Iterations" in Advanced Configuration (try 15-25)

2. **Agent Takes Too Long**
   - Increase timeout in Advanced Configuration
   - Enable Debug Mode to see what's happening
   - Try simpler queries first

3. **Agent Gives Incomplete Answers**
   - Increase max_iterations (complex queries need more steps)
   - Check if ChromaDB has sufficient data
   - Enable verbose logging to see tool calls

### Common Issues

4. **CUDA Out of Memory**
   ```bash
   # Use CPU-only mode or enable 4-bit quantization
   python streamlit_agent.py --use-cpu
   ```

5. **API Key Issues**
   ```bash
   # Test API key validity
   python -c "from gemini_llm import GeminiLLM; GeminiLLM.test_connection('your_key')"
   ```

6. **ChromaDB Not Found**
   ```bash
   # Regenerate vector database
   python migrate_to_chromadb.py
   ```

7. **Import Errors**
   ```bash
   # Use module syntax for tests
   python -m test.run_all_tests
   ```

### Performance Optimization

- **GPU Acceleration**: Enable CUDA for local models
- **Model Quantization**: Use 4-bit precision for memory efficiency
- **Caching**: Enable response caching for repeated queries
- **Batch Processing**: Process multiple queries together

## 📚 Documentation

- **[Hybrid LLM Setup Guide](HYBRID_LLM_SETUP.md)**: Detailed configuration guide
- **[API Documentation](docs/api.md)**: Complete API reference
- **[Tool Development](docs/tools.md)**: Guide for creating custom tools

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🎯 Use Cases

### Business Owners
- **Customer Feedback Analysis**: Understand what customers really think
- **Competitor Research**: Analyze competitor reviews and positioning
- **Market Insights**: Identify trends and opportunities

### Data Analysts
- **Sentiment Tracking**: Monitor sentiment changes over time
- **Aspect Analysis**: Deep dive into specific business aspects
- **Performance Benchmarking**: Compare against industry standards

### Developers
- **Custom Tool Development**: Build specialized analysis tools
- **API Integration**: Integrate with existing business systems
- **Scalable Deployment**: Deploy in cloud or on-premises

## 🌟 Future Roadmap

- **🔄 Multi-Model Support**: Integration with OpenAI, Anthropic, and other providers
- **📱 Mobile App**: Native mobile interface
- **🔄 Real-time Processing**: Live review analysis and alerts
- **🤖 AutoML**: Automated model selection and optimization
- **📊 Advanced Visualizations**: Interactive dashboards and reports

---

## 🚀 Get Started Now!

```bash
# One-command setup
conda env create -f environment-hybrid.yml && conda activate hybrid-llm-bi && streamlit run streamlit_agent.py
```

**Experience the future of business intelligence with hybrid AI models!** 🎉

---

*Built with ❤️ using LangChain, Streamlit, ChromaDB, and cutting-edge AI models*