import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json


class ActionPlannerTool:
    """
    Tool for generating actionable business improvement plans.
    Follows the same modular pattern as other tools in the system.
    """

    def __init__(self, data_path: str = "data/processed/review_cleaned.parquet"):
        """Initialize with review data - consistent with DataSummaryTool pattern"""
        self.df = pd.read_parquet(data_path) if data_path.endswith('.parquet') else pd.read_csv(data_path)

        # Action templates for common business improvements
        self.action_templates = self._initialize_action_templates()

    def _initialize_action_templates(self) -> Dict[str, Dict[str, Any]]:
        """Initialize action templates - modular configuration like BusinessSearchTool"""
        return {
            "quality": {
                "title": "Implement comprehensive quality improvement program",
                "owner": "Quality Manager",
                "base_steps": [
                    "Conduct quality audit and identify improvement areas",
                    "Develop quality standards and procedures",
                    "Train staff on quality protocols",
                    "Implement quality monitoring system"
                ],
                "required_resources": ["Quality assessment tools", "Training materials", "Monitoring systems"],
                "risk": "Staff resistance to new quality standards",
                "mitigation": "Involve staff in standards development and provide clear benefits explanation",
                "target_metrics": ["quality_mentions", "avg_rating"]
            },
            "service": {
                "title": "Enhance customer service excellence program",
                "owner": "Customer Experience Manager",
                "base_steps": [
                    "Assess current service gaps through customer feedback",
                    "Develop service standards and training curriculum",
                    "Implement customer service training program",
                    "Establish service quality monitoring and feedback loop"
                ],
                "required_resources": ["Service training program", "Customer feedback system", "Performance metrics"],
                "risk": "Inconsistent service delivery across staff",
                "mitigation": "Regular coaching sessions and performance incentives",
                "target_metrics": ["service_satisfaction", "customer_complaints"]
            },
            "value": {
                "title": "Optimize value proposition and pricing strategy",
                "owner": "Business Development Manager",
                "base_steps": [
                    "Analyze competitor pricing and value offerings",
                    "Review current pricing strategy and customer value perception",
                    "Develop enhanced value proposition",
                    "Implement pricing adjustments or value-added services"
                ],
                "required_resources": ["Market research", "Pricing analysis tools", "Value enhancement options"],
                "risk": "Price changes may affect customer retention",
                "mitigation": "Gradual implementation with clear communication of added value",
                "target_metrics": ["value_perception", "price_satisfaction"]
            },
            "customer_experience": {
                "title": "Transform overall customer experience journey",
                "owner": "Customer Experience Director",
                "base_steps": [
                    "Map current customer journey and identify pain points",
                    "Design improved customer experience touchpoints",
                    "Implement customer experience improvements",
                    "Monitor and optimize customer satisfaction metrics"
                ],
                "required_resources": ["Customer journey mapping", "Experience design tools", "Satisfaction tracking"],
                "risk": "Complex changes may create temporary disruption",
                "mitigation": "Phased rollout with continuous customer communication",
                "target_metrics": ["customer_satisfaction", "net_promoter_score"]
            }
        }

    def _calculate_baseline_metrics(self, business_id: str) -> Dict[str, Any]:
        """Calculate baseline metrics - similar to DataSummaryTool approach"""
        if business_id is None:
            business_reviews = self.df
        else:
            business_reviews = self.df[self.df['business_id'] == business_id]

        if business_reviews.empty:
            return {}

        total_reviews = len(business_reviews)
        avg_rating = business_reviews['stars'].mean()

        # Calculate general issue mention rates across industries
        quality_issues = business_reviews['text'].str.contains(
            'quality|poor|bad|terrible|awful|disappointing', case=False, na=False
        ).sum()
        service_issues = business_reviews['text'].str.contains(
            'service|staff|rude|unfriendly|helpful|polite', case=False, na=False
        ).sum()
        value_issues = business_reviews['text'].str.contains(
            'expensive|cheap|price|cost|value|money|overpriced', case=False, na=False
        ).sum()

        return {
            "business_id": business_id or "all businesses",
            "total_reviews": total_reviews,
            "avg_rating": round(avg_rating, 2),
            "quality_mentions_rate": round((quality_issues / total_reviews) * 100, 1),
            "service_mentions_rate": round((service_issues / total_reviews) * 100, 1),
            "value_mentions_rate": round((value_issues / total_reviews) * 100, 1)
        }

    def _generate_action(self, issue_type: str, baseline: Dict[str, Any], constraints: Dict[str, Any],
                         action_id: int) -> Dict[str, Any]:
        """Generate single action - modular helper method"""
        if issue_type not in self.action_templates:
            return {}

        template = self.action_templates[issue_type].copy()
        weeks = constraints.get('timeline_weeks', 8)
        deadline = (datetime.now() + timedelta(weeks=weeks)).strftime('%Y-%m-%d')

        # Create action based on template
        action = {
            "id": f"A{action_id}",
            "title": template["title"],
            "owner": template["owner"],
            "deadline": deadline,
            "steps": template["base_steps"].copy(),
            "required_resources": template["required_resources"].copy(),
            "risk": template["risk"],
            "mitigation": template["mitigation"]
        }

        # Add KPIs based on target metrics
        kpis = {}
        for metric in template["target_metrics"]:
            if metric == "quality_mentions":
                kpis[metric] = {"target": f"-50% in {weeks} weeks"}
            elif metric == "avg_rating":
                kpis[metric] = {"target": f"+0.4 in {weeks} weeks"}
            else:
                kpis[metric] = {"target": f"+30% in {weeks} weeks"}

        action["kpis"] = kpis

        # Add evidence section
        baseline_rate = baseline.get(f"{issue_type}_mentions_rate", 0)
        action["evidence"] = {
            "why": [
                {"reason": f"{issue_type} identified as priority issue", "source": "BusinessAnalysis"},
                {"reason": f"Current {issue_type} mention rate: {baseline_rate}%", "source": "DataSummary"}
            ],
            "baseline": {
                f"{issue_type}_mentions_rate": f"{baseline_rate}%",
                "avg_rating": f"{baseline.get('avg_rating', 0)}"
            }
        }

        return action

    def _generate_roadmap(self, actions: List[Dict[str, Any]], timeline_weeks: int) -> List[Dict[str, Any]]:
        """Generate implementation roadmap - similar to structured output of other tools"""
        if not actions:
            return []

        roadmap = []
        weeks_per_action = max(1, timeline_weeks // len(actions))

        for week in range(1, timeline_weeks + 1):
            focus_areas = []

            for i, action in enumerate(actions):
                action_start_week = (i * weeks_per_action) + 1
                action_end_week = min(action_start_week + weeks_per_action - 1, timeline_weeks)

                if action_start_week <= week <= action_end_week:
                    steps = action.get('steps', [])
                    if steps:
                        step_index = min((week - action_start_week), len(steps) - 1)
                        focus_areas.append(f"{action['id']}: {steps[step_index]}")

            if focus_areas:
                roadmap.append({"week": week, "focus": focus_areas})

        return roadmap

    def _estimate_costs(self, actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Estimate implementation costs - helper method like other tools"""
        total_cost = 0
        cost_breakdown = {}

        for action in actions:
            resources = action.get('required_resources', [])
            action_cost = len(resources) * 1000  # Simplified cost model
            total_cost += action_cost
            cost_breakdown[action['id']] = action_cost

        return {
            "total_estimated_cost": total_cost,
            "cost_breakdown": cost_breakdown,
            "cost_per_week": total_cost // max(1, len(actions) * 2)  # Rough weekly estimate
        }

    def __call__(self, business_id: Optional[str] = None, goals: List[str] = None,
                 constraints: Dict[str, Any] = None, priority_issues: List[str] = None) -> Dict[str, Any]:
        """
        Main callable method - follows same pattern as other tools
        Simple interface with optional parameters like DataSummaryTool
        """
        # Default parameters
        if goals is None:
            goals = ["improve_customer_satisfaction"]
        if constraints is None:
            constraints = {"budget": 10, "timeline_weeks": 1}
        if priority_issues is None:
            priority_issues = ["quality", "service"]

        # Calculate baseline metrics - same pattern as DataSummaryTool
        baseline = self._calculate_baseline_metrics(business_id)

        if not baseline:
            return {
                "business_id": business_id,
                "error": f"No data found for business_id: {business_id}",
                "actions": [],
                "roadmap": [],
                "cost_analysis": {}
            }

        # Generate actions for priority issues
        actions = []
        for i, issue in enumerate(priority_issues, 1):
            if issue in self.action_templates:
                action = self._generate_action(issue, baseline, constraints, i)
                if action:
                    actions.append(action)

        # Generate roadmap and cost analysis
        timeline_weeks = constraints.get('timeline_weeks', 8)
        roadmap = self._generate_roadmap(actions, timeline_weeks)
        cost_analysis = self._estimate_costs(actions)

        # Calculate success probability (simplified model)
        base_probability = 0.6
        action_bonus = min(0.3, len(actions) * 0.1)
        budget_factor = min(1.0, constraints.get('budget', 5000) / 5000)
        success_probability = min(0.95, base_probability + action_bonus * budget_factor)

        # Return structured result - consistent with other tools
        return {
            "business_id": business_id,
            "goals": goals,
            "constraints": constraints,
            "baseline_metrics": baseline,
            "actions": actions,
            "roadmap": roadmap,
            "cost_analysis": cost_analysis,
            "success_probability": round(success_probability, 2),
            "implementation_summary": {
                "total_actions": len(actions),
                "implementation_weeks": timeline_weeks,
                "average_cost_per_action": cost_analysis.get("total_estimated_cost", 0) // max(1, len(actions)),
                "complexity_level": "high" if len(actions) > 3 else "medium" if len(actions) > 1 else "low"
            }
        }