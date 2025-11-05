install environments

```bash
conda env create -f langchain-demo-env.yml -y
```

activate environment

```bash
conda activate langchain-demo
```

install pytorch based on your version following this link https://pytorch.org/get-started/locally/


if you want to preprocessing and embeddings
first, make sure that the yelp dataset is in the correct location, which is raw_data folder, then run preprocessing
```bash
cd scripts
python run_preprocessing.py --sample-size <sample size under 7 millions>
```
then run embeddings
```bash
python migrate_to_chromadb.py
python migrate_business_to_chromadb.py
```
then to create duckdb embedded database
```bash
cd migration
python setup_database.py
```

to start the Chroma Server
```bash
cd scripts
python start_chroma_servers.py
```

to setup the .env file
use the .env.sample file to create .env file

finally, to run the actual system
```bash
streamlit run streamlit_agent.py
```

to run the behavior evaluation
```bash
python behavior_langsmith_eval.py 
```
to run the capability evaluation
```bash
python capabilities_langsmith_eval.py
```

to run the LLM judge for agent answer
```bash
python simple_langsmith_eval_with_agent.py
```
