# Business Recommendation Agent

This project contains a business recommendation agent that can suggest businesses based on user queries.

## Available Versions

- **Original Agent** (`demo.py`, `demo2.py`) - Custom ReAct implementation
- **LangChain Agent** (`demo2_langchain.py`) - Enhanced version using LangChain framework ✨ **NEW**

## Setup

There are two main parts of this project: data preprocessing and running the agent.

### For Data Preprocessing

If you want to preprocess the data from scratch, follow these steps.

1. **Install dependencies:**
   ```bash
   pip install -r requirements-minimal.txt
   ```
2. **Run the preprocessing script:**
   This will clean the raw data. The output will be in `data/processed`.
   ```bash
   python scripts/run_preprocessing.py
   ```
3. **Build the search index:**
   This will create an index from the processed data for the search tool.
   ```bash
   python build_index_from_csv.py
   ```

### For Running the Agent

If you want to run the agent with preprocessed data and a pre-built index, follow these steps.

1. **Download pre-built data and index:**

   * Download the `index_demo` folder from [Google Drive](https://drive.google.com/drive/folders/1Y4gnvplLDlb5-wxB2W3M4QbQFi8mhrWL?usp=sharing) and place it in the project root.
   * Download the `processed` data folder from [Google Drive](https://drive.google.com/drive/folders/1n2D1Cq0MhgSDKI55GOGQ4btO1p_A4RzV?usp=sharing) and place it inside the `data/processed` folder.
2. **Create and activate Conda environment:**

   ```bash
   conda env create -f another_env.yml
   conda activate biz-agent-gpu-2
   ```
3. **Download language model:**

   ```bash
   python -m spacy download en_core_web_sm
   ```
4. **(Optional) For NVIDIA GPU users:**
   If you have a CUDA-enabled GPU, install the appropriate PyTorch version:

   ```bash
   pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118
   ```
5. **Run the demo:**

   ```bash
   python demo2.py
   ```

### For LangChain Agent (Recommended) ✨

The LangChain version provides enhanced capabilities and better error handling:

#### Quick Start

1. **Download data** (same as above):
   * Download the `processed` data folder from [Google Drive](https://drive.google.com/drive/folders/1n2D1Cq0MhgSDKI55GOGQ4btO1p_A4RzV?usp=sharing) and place it inside the `data` folder.

2. **Create environment** (choose one):

   **For GPU users:**
   ```bash
   conda env create -f langchain-demo-env.yml
   conda activate langchain-demo
   ```

   **For CPU-only users:**
   ```bash
   conda env create -f langchain-demo-cpu.yml
   conda activate langchain-demo-cpu
   ```

3. **Run the LangChain demo:**
   ```bash
   python demo2_langchain.py
   ```

#### Features
- 🔧 **Better Tool Integration**: Proper schema validation and error handling
- 🧠 **Memory Management**: Conversation memory for multi-turn interactions  
- 📊 **Enhanced Monitoring**: Detailed execution logging and debugging
- 🔄 **Robust Architecture**: Battle-tested LangChain framework
- ⚡ **Performance**: Optimized execution with fallback options

#### Troubleshooting
For detailed setup instructions and troubleshooting, see [SETUP_LANGCHAIN.md](SETUP_LANGCHAIN.md).
