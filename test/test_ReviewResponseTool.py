import sys
import os
import json

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from tools.ReviewResponseTool import ReviewResponseTool


def test_review_response_llm_io():
    """Test ReviewResponseTool LLM input/output interface - showing only essential data flow"""

    print("=== ReviewResponseTool LLM Input/Output Test ===\n")

    # Initialize tool
    tool = ReviewResponseTool(
        business_data_path="data/processed/business_cleaned.parquet",
        review_data_path="data/processed/review_cleaned.parquet"
    )

    # === LLM INPUT TO REVIEWRESPONSE ===
    print("INPUT from LLM to ReviewResponseTool:")
    print("-" * 40)

    llm_input = {
        "business_id": "XQfwVwDr-v0ZS3_CbbE5Xw",
        "review_text": "The food was terrible and the service was incredibly slow. We waited 45 minutes for our order and when it finally came, it was cold. The staff seemed rude and uninterested. Never coming back!",
        "response_tone": "professional"
    }

    print(json.dumps(llm_input, indent=2))

    # === CALL THE TOOL ===
    result = tool(
        business_id=llm_input["business_id"],
        review_text=llm_input["review_text"],
        response_tone=llm_input["response_tone"]
    )

    # === LLM OUTPUT FROM REVIEWRESPONSE ===
    print("\n" + "=" * 50)
    print("OUTPUT from ReviewResponseTool to LLM:")
    print("-" * 40)

    # Clean output - only essential fields for LLM
    clean_output = {
        "business_id": result.get("business_id"),
        "review_analysis": result.get("review_analysis", {}),
        "generated_response": result.get("generated_response", ""),
        "response_tone": result.get("response_tone"),
        "key_points_addressed": result.get("key_points_addressed", []),
        "escalation_flags": result.get("escalation_flags", []),
        "response_metadata": _extract_response_metadata(result.get("response_metadata", {}))
    }

    print(json.dumps(clean_output, indent=2))

    # === VALIDATION ===
    print("\n" + "=" * 50)
    print("VALIDATION:")
    print("-" * 40)

    # Check input was processed
    assert result.get("business_id") == llm_input["business_id"], "Business ID mismatch"
    assert result.get("response_tone") == llm_input["response_tone"], "Response tone mismatch"

    if "error" in result:
        print(f"❌ FAILED: {result['error']}")
        return False
    else:
        print("✅ INPUT: Valid review response request received")
        print("✅ OUTPUT: Complete response generated")

        # Check review analysis
        analysis = result.get("review_analysis", {})
        if analysis:
            sentiment = analysis.get("sentiment", "unknown")
            issues = analysis.get("specific_issues", [])
            urgency = analysis.get("urgency_level", "unknown")

            print(f"✅ ANALYSIS: {sentiment} sentiment detected")
            print(f"✅ ISSUES: {len(issues)} specific issues identified: {', '.join(issues) if issues else 'none'}")
            print(f"✅ URGENCY: {urgency} priority level")

        # Check generated response
        response = result.get("generated_response", "")
        if response:
            word_count = len(response.split())
            print(f"✅ RESPONSE: {word_count} words generated")

        # Check key points
        key_points = result.get("key_points_addressed", [])
        print(f"✅ KEY POINTS: {len(key_points)} points to address")

        # Check escalation flags
        flags = result.get("escalation_flags", [])
        if flags:
            print(f"⚠️  ESCALATION: {len(flags)} flags detected: {', '.join(flags)}")
        else:
            print("✅ ESCALATION: No escalation required")

        # Check metadata
        metadata = result.get("response_metadata", {})
        if metadata:
            requires_review = metadata.get("requires_review", False)
            followup = metadata.get("suggested_followup", "none")
            print(f"✅ METADATA: Review required: {requires_review}, Followup: {followup}")

    return True


def _extract_response_metadata(metadata):
    """Extract essential metadata for LLM"""
    if not metadata:
        return {}

    return {
        "character_count": metadata.get("character_count", 0),
        "requires_review": metadata.get("requires_review", False),
        "suggested_followup": metadata.get("suggested_followup", "none")
    }


def test_different_review_scenarios():
    """Test different types of reviews and response tones"""

    print("\n" + "=" * 60)
    print("TESTING DIFFERENT REVIEW SCENARIOS:")
    print("=" * 60)

    tool = ReviewResponseTool(
        business_data_path="data/processed/business_cleaned.parquet",
        review_data_path="data/processed/review_cleaned.parquet"
    )

    # Test scenarios with different sentiments and tones
    test_scenarios = [
        {
            "name": "Positive Review - Friendly Tone",
            "input": {
                "business_id": "XQfwVwDr-v0ZS3_CbbE5Xw",
                "review_text": "Amazing food and excellent service! The staff was so friendly and the atmosphere was perfect. Will definitely come back!",
                "response_tone": "friendly"
            }
        },
        {
            "name": "Neutral Review - Professional Tone",
            "input": {
                "business_id": "XQfwVwDr-v0ZS3_CbbE5Xw",
                "review_text": "The food was okay and the service was decent. Nothing special but not bad either. Average experience overall.",
                "response_tone": "professional"
            }
        },
        {
            "name": "Severe Complaint - Formal Tone",
            "input": {
                "business_id": "XQfwVwDr-v0ZS3_CbbE5Xw",
                "review_text": "This place made me sick! I got food poisoning and had to go to the hospital. The health department needs to investigate this place!",
                "response_tone": "formal"
            }
        },
        {
            "name": "Empty Review Text",
            "input": {
                "business_id": "XQfwVwDr-v0ZS3_CbbE5Xw",
                "review_text": "",
                "response_tone": "professional"
            }
        }
    ]

    for scenario in test_scenarios:
        print(f"\n--- {scenario['name']} ---")

        result = tool(**scenario["input"])

        if "error" in result:
            print(f"❌ {scenario['name']}: {result['error']}")
        else:
            analysis = result.get("review_analysis", {})
            sentiment = analysis.get("sentiment", "unknown")
            response_length = len(result.get("generated_response", "").split())
            urgency = analysis.get("urgency_level", "unknown")

            print(f"✅ {scenario['name']}")
            print(f"   Sentiment: {sentiment}")
            print(f"   Urgency Level: {urgency}")
            print(f"   Response Length: {response_length} words")

            # Show sample of generated response
            response = result.get("generated_response", "")
            if response:
                sample = response[:100] + "..." if len(response) > 100 else response
                print(f"   Response Sample: \"{sample}\"")


def test_response_tone_variations():
    """Test the same review with different response tones"""

    print("\n" + "=" * 60)
    print("TESTING RESPONSE TONE VARIATIONS:")
    print("=" * 60)

    tool = ReviewResponseTool(
        business_data_path="data/processed/business_cleaned.parquet",
        review_data_path="data/processed/review_cleaned.parquet"
    )

    base_review = "The food was good but the service was a bit slow. Overall an okay experience."
    tones = ["professional", "friendly", "formal"]

    print(f"Base Review: \"{base_review}\"\n")

    for tone in tones:
        result = tool(
            business_id="Pns2l4eNsfO8kk83dixA6A",
            review_text=base_review,
            response_tone=tone
        )

        response = result.get("generated_response", "")
        print(f"--- {tone.upper()} TONE ---")
        print(f"Response: \"{response}\"")
        print(f"Length: {len(response.split())} words")
        print()


if __name__ == "__main__":
    # Run main test
    success = test_review_response_llm_io()

    if success:
        # Run scenario tests
        test_different_review_scenarios()

        # Run tone variation tests
        test_response_tone_variations()

        print(f"\n{'=' * 60}")
        print("ReviewResponseTool test completed successfully!")
    else:
        print("ReviewResponseTool test failed!")