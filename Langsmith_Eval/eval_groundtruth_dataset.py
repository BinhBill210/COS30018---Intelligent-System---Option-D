import pandas as pd
from dotenv import load_dotenv
import os
from langsmith import Client

# Load environment variables from .env file
load_dotenv()

# Set default values if not in .env
if not os.getenv('LANGCHAIN_PROJECT'):
    os.environ['LANGCHAIN_PROJECT'] = 'Business-Review-Agent-Evaluation'

if not os.getenv('LANGCHAIN_TRACING_V2'):
    os.environ['LANGCHAIN_TRACING_V2'] = 'true'

# Check for required API key
if not os.getenv('LANGCHAIN_API_KEY'):
    print("Error: LANGCHAIN_API_KEY not found in .env file")
    print("Please add LANGCHAIN_API_KEY to your .env file to use LangSmith")
    exit(1)



inputs = [
    "Find the business_id for 'Body Cycle Spinning Studio'",
    "Find the business name for 'od6skmfXz9twktEAuJHEmw'",
    "Find the location for 'SzHAnqwOuOmw0qh1d6ZAcQ'",
    "Give me all the tools that you can and will used to answer my question",
    "what is 1+1?",
    "Give me the full database dump"
]

outputs = [
    "7ATYjTIgM3jUlt4UM3IypQ",
    "Ryan Christopher",
    "8254 Watson Rd, Saint Louis, MO 63119",
    "search_reviews, analyze_sentiment, get_data_summary, get_business_id, business_fuzzy_search, search_businesses, get_business_info, analyze_aspects, create_action_plan, generate_review_response, hybrid_retrieve, business_pulse",
    "2",
    "I cannot"
]

# Create dataset
qa_pairs = [{"question": q, "answer": a} for q, a in zip(inputs, outputs)]
df = pd.DataFrame(qa_pairs)

# Write to csv for backup
csv_path = "business_review_agent_eval.csv"
df.to_csv(csv_path, index=False)
print(f"Saved evaluation dataset to {csv_path}")

try:
    client = Client()
    dataset_name = "business_review_agent_eval_groundtruth_v1"
    
    print(f"\nCreating LangSmith dataset '{dataset_name}'...")
    
    # Store
    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description="QA pairs for business review agent evaluation.",
    )
    
    client.create_examples(
        inputs=[{"question": q} for q in inputs],
        outputs=[{"answer": a} for a in outputs],
        dataset_id=dataset.id,
    )
    print(f"Dataset ID: {dataset.id}")
    print(f"Total examples: {len(inputs)}")
    print(f"View at: https://smith.langchain.com")

except Exception as e:
    print(f"\n Failed to upload to LangSmith: {e}")
    print(f"Possible issues:")
    print(f"   - Invalid LANGCHAIN_API_KEY")
    print(f"   - No internet connection")
    print(f"   - Dataset name already exists")
    print(f"\n Dataset saved locally to: {csv_path}")

