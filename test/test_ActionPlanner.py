from dotenv import load_dotenv
load_dotenv()
import sys
import os
import json

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from tools.ActionPlanner import ActionPlannerTool


def test_action_planner_llm_io():
    """Test ActionPlanner LLM input/output interface - showing only essential data flow"""

    print("=== ActionPlanner LLM Input/Output Test ===\n")

    # Initialize tool
    tool = ActionPlannerTool("data/processed/review_cleaned.parquet")

    # === LLM INPUT TO ACTIONPLANNER ===
    print("INPUT from LLM to ActionPlanner:")
    print("-" * 40)

    llm_input = {
        "business_id": "XQfwVwDr-v0ZS3_CbbE5Xw",
        "goals": ["improve_customer_satisfaction"],
        "constraints": {"budget": 10, "timeline_weeks": 1},
        "priority_issues": ["quality", "service"]
    }

    print(json.dumps(llm_input, indent=2))

    # Show available templates
    print(f"\nAvailable action templates: {list(tool.action_templates.keys())}")

    # === CALL THE TOOL ===
    result = tool(
        business_id=llm_input["business_id"],
        goals=llm_input["goals"],
        constraints=llm_input["constraints"],
        priority_issues=llm_input["priority_issues"]
    )

    # === LLM OUTPUT FROM ACTIONPLANNER ===
    print("\n" + "=" * 50)
    print("OUTPUT from ActionPlanner to LLM:")
    print("-" * 40)

    # Clean output - only essential fields for LLM
    clean_output = {
        "business_id": result.get("business_id"),
        "baseline_metrics": result.get("baseline_metrics", {}),
        "actions": _extract_action_summaries(result.get("actions", [])),
        "roadmap_summary": _extract_roadmap_summary(result.get("roadmap", [])),
        "cost_analysis": result.get("cost_analysis", {}),
        "success_probability": result.get("success_probability"),
        "implementation_summary": result.get("implementation_summary", {})
    }

    print(json.dumps(clean_output, indent=2))

    # === VALIDATION ===
    print("\n" + "=" * 50)
    print("VALIDATION:")
    print("-" * 40)

    # Check input was processed
    assert result.get("business_id") == llm_input["business_id"], "Business ID mismatch"

    if "error" in result:
        print(f"❌ FAILED: {result['error']}")
        return False
    else:
        print("✅ INPUT: Valid JSON structure received")
        print("✅ OUTPUT: Complete action plan generated")

        # Debug action generation
        expected_actions = len(llm_input["priority_issues"])
        actual_actions = len(result.get("actions", []))
        print(f"Expected actions: {expected_actions} (for issues: {llm_input['priority_issues']})")
        print(f"Actual actions: {actual_actions}")

        # Show which templates are available
        available_templates = list(tool.action_templates.keys())
        print(f"Available templates: {available_templates}")

        print("Checking which priority issues matched templates:")
        for issue in llm_input["priority_issues"]:
            if issue in available_templates:
                print(f"  ✅ '{issue}' - template found")
            else:
                print(f"  ❌ '{issue}' - no template found")

        if actual_actions > 0:
            print("Actions generated:")
            for i, action in enumerate(result.get("actions", [])):
                print(f"  Action {i + 1}: {action.get('title', 'No title')} (ID: {action.get('id')})")

        # Show roadmap details
        roadmap_info = clean_output.get("roadmap_summary", {})
        total_weeks = roadmap_info.get("total_weeks", 0)
        print(f"✅ ROADMAP: {total_weeks} weeks planned")

        print(f"✅ COST: ${result.get('cost_analysis', {}).get('total_estimated_cost', 0)} estimated")

    return True


def _extract_action_summaries(actions):
    """Extract essential action info for LLM"""
    summaries = []
    for action in actions:
        summary = {
            "id": action.get("id"),
            "title": action.get("title"),
            "owner": action.get("owner"),
            "deadline": action.get("deadline"),
            "steps_count": len(action.get("steps", [])),
            "key_kpis": list(action.get("kpis", {}).keys()),
            "risk": action.get("risk")
        }
        summaries.append(summary)
    return summaries


def _extract_roadmap_summary(roadmap):
    """Extract complete roadmap for LLM"""
    if not roadmap:
        return {"total_weeks": 0, "all_weeks": [], "focus_areas": 0}

    return {
        "total_weeks": len(roadmap),
        "all_weeks": roadmap,  # Show complete roadmap, not just samples
        "focus_areas": len([item for week in roadmap for item in week.get("focus", [])])
    }


if __name__ == "__main__":
    test_action_planner_llm_io()