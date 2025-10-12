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
    "Give me all the review about Body Cycle Spinning Studio",
    "Give me a summary about Body Cycle Spinning Studio",
    "Give me all the tools that you can and will used to answer my question",
    "list all the restaurant with 'vietnamese' in it's name",
    "what is 1+1?",
    "Give me the full database dump"
]

outputs = [
    "7ATYjTIgM3jUlt4UM3IypQ",
    "Ryan Christopher",
    "8254 Watson Rd, Saint Louis, MO 63119",
    """
        Review 1: ". Carly was happy to help my new-to-cycling sister and I in adjusting our bikes, and explaining all of the terminology she would be using during her class. She was particularly mindful of my injuries (lumbar back spasm and degeneration of left knee cartilage) as she adjusted me into my bike, and reminded me to not forget that this was "my ride" and to listen to my body, above all. The next 30 minutes was a blur of sweat, music, encouragement, challenge, and accomplishment" - Stars: 5.0 - Date: 2013-03-07
        Review 2: "Great class with Russell who really made an effort to make sure every cyclist new and regular was properly set up on their bikes. Also showed and reminded everyone of proper form. Bring your own shoes or cycle in your sneakers." - Stars: 5.0 - Date: 2015-07-12
        Review 3: ". What's more, when I returned two days later, he saw my name on the list and came in to personally apologize for the mix up. He truly went above and beyond to make me feel like a valued customer. When I went to class this morning, Russell was taking customers' coats as they arrived and hanging them up. I've never seen such a thing at a fitness studio! At the end of class, Russell was waiting with a mop to clean up all of the sweat we generated during an intense ride" - Stars: 5.0 - Date: 2012-12-28
        Review 4: "This place is awesome! A friend dragged me here after spotting an event on facebook called B cubed (B^3) = Bikes.Bootcamp.Brunch. For $25 you got a 30 minute spin session (courtesy of Russell who is also the owner and is fantastic!), a run to the art museum (which I skipped and drove to the art museum cause I parked on the street, oh, and I also hate running!)" - Stars: 5.0 - Date: 2016-06-26
        Review 5: ". Everyone there is super helpful and all give amazing customer service. I'm a fan, can you tell." - Stars: 5.0 - Date: 2015-07-14
    """,
    """
            Total Reviews: 147
            Average Stars: 4.75
            Average Useful: 1.38
            Average Funny: 0.37
            Average Cool: 0.7
    """,
    "search_reviews, analyze_sentiment, get_data_summary, get_business_id, business_fuzzy_search, search_businesses, get_business_info, analyze_aspects, create_action_plan, generate_review_response, hybrid_retrieve, business_pulse",
    "Vina Vietnamese Restaurant, Chao Vietnamese Street Food, Mekong Vietnamese Restaurant, Saigon Quy-Bau Restaurant, Saigon Vietnamese Restaurant",
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

