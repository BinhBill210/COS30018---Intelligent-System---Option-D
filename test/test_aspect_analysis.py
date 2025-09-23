from dotenv import load_dotenv
load_dotenv()


import json
import sys
from tools.aspect_analysis import AspectABSAToolHF

def test_aspect_absa_tool():
    print("Testing AspectABSAToolHF...")

 
    business_id ="XQfwVwDr-v0ZS3_CbbE5Xw"
    print(f"- business_id: {business_id}")

    tool = AspectABSAToolHF()

    reviews = tool.read_data(business_id=business_id)
    print(f"- Loaded {len(reviews)} reviews")

    result = tool.analyze_aspects(reviews)
    aspects = result.get("aspects", {})

   
    print("Output (aspects):")
    print(json.dumps(aspects, ensure_ascii=False, indent=2))

    # condition pass: has at least 1 aspect
    ok = isinstance(aspects, dict) and len(aspects) > 0
    print(f"Test passed: {ok}")
    return ok

if __name__ == "__main__":
    test_aspect_absa_tool()