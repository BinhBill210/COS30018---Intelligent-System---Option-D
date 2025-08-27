# COS30018 Intelligent System - Option D

## 🛠️ Environment Setup

1. **Install Anaconda Environment**
   ```bash
   conda env create -f langchain-demo-env.yml
   ```
   For NVIDIA GPUs with CUDA support, install PyTorch according to your CUDA version:
   - Visit [PyTorch Get Started](https://pytorch.org/get-started/locally/) for instructions.

---

## 📦 Data Preprocessing

- Run preprocessing from the project root:
  ```bash
  python scripts/run_preprocessing.py
  ```
- For argument details:
  ```bash
  python scripts/run_preprocessing.py --help
  ```
- **Alternative:**
  - Download the `data` folder from [Google Drive](https://drive.google.com/drive/u/0/folders/1enrB0_dKmCJG62NjTBqRG_pZF76Xv4z9) and place it in the project root.

---

## 🔗 Review Embeddings

- Run embedding migration:
  ```bash
  python migrate_to_chromadb.py
  ```
- **Alternative:**
  - Download the `chroma_db` folder from [Google Drive](https://drive.google.com/drive/u/0/folders/1enrB0_dKmCJG62NjTBqRG_pZF76Xv4z9) and place it in the project root.

---

## 🧪 Tool Testing

- Run individual tool tests or all tests:
  - **Recommended:** Run all tests as a module to avoid Python path issues:
    ```bash
    python -m test.run_all_tests
    ```
  - Or run individual tests:
    ```bash
    python -m test.test_review_search_tool
    python -m test.test_sentiment_summary_tool
    python -m test.test_data_summary_tool
    ```

---

## 🤖 Model Demo

- Run the LangChain agent demo:
  ```bash
  python langchain_agent_chromadb.py
  ```
  This will show example queries and responses using the integrated tools and ChromaDB backend.

---

## 📚 Additional Notes

- Ensure you run all commands from the project root directory.
- For troubleshooting Python import errors, always use the `-m` module syntax for tests.
- For GPU acceleration, confirm your CUDA version and install compatible PyTorch.

---

**For more details, see the main `readme.md` and setup guides in the repository.**
