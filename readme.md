install environments

```bash
conda env create -f langchain-demo-env.yml -y
```

activate environment

```bash
conda activate langchain-demo
```

install pytorch based on your version following this link https://pytorch.org/get-started/locally/

manually install streamlit
```bash
pip install streamlit
```


1M dataset, chroma, duckdb files: https://drive.google.com/drive/u/0/folders/1enrB0_dKmCJG62NjTBqRG_pZF76Xv4z9

full dataset, chroma, duckdb files: https://liveswinburneeduau-my.sharepoint.com/my?id=%2Fpersonal%2F104788737%5Fstudent%5Fswin%5Fedu%5Fau%2FDocuments%2FFSwinWork%2FFull%20Dataset

.env.sample is there, put the values in the .env file accordingly

run python -m test.run_all.tests to test the tools

run the langchain agent with streamlit

```bash
streamlit run streamlit_agent.py
```