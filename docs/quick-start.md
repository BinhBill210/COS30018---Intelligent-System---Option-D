# 🚀 Quick Start Guide

Get up and running with the Hybrid LLM Business Intelligence System in minutes!

## ⚡ Express Setup (3 minutes)

### 1. Environment Setup
```bash
# Clone and setup
git clone <your-repo-url>
cd COS30018---Intelligent-System---Option-D

# One command setup
conda env create -f environment-hybrid.yml
conda activate hybrid-llm-bi
```

### 2. Get Data (Choose One)
**Fast way**: Download from [Google Drive](https://drive.google.com/drive/u/0/folders/1enrB0_dKmCJG62NjTBqRG_pZF76Xv4z9)
- Download `data` folder → place in project root
- Download `chroma_db` folder → place in project root

**Build yourself**:
```bash
python scripts/run_preprocessing.py
python migrate_to_chromadb.py
```

### 3. Launch
```bash
streamlit run streamlit_agent.py
```
Open `http://localhost:8501` 🎉

## 🎯 First Steps

1. **Try Local Model First**
   - Select "local" in Model Configuration
   - Click "Load Agent"
   - Ask: *"What are customers saying about food quality?"*

2. **Add Gemini Power** (Optional)
   - Get API key from [Google AI Studio](https://aistudio.google.com/app/apikey)
   - Use "API Key Management" in sidebar
   - Select "gemini" model and reload

3. **Go Hybrid**
   - Select "hybrid" mode
   - Choose primary/fallback models
   - Enjoy automatic fallback!

## 🆘 Need Help?

- **Can't load models?** → Check GPU memory, try CPU mode
- **API errors?** → Verify API key in "API Key Management"
- **Import errors?** → Run `python -m test.run_all_tests`
- **Missing data?** → Download from Google Drive link above

## 📖 Next Steps

- Read the [full README](../README.md) for detailed features
- Check [HYBRID_LLM_SETUP.md](../HYBRID_LLM_SETUP.md) for advanced config
- Explore the business intelligence tools and analytics

Happy analyzing! 🚀
