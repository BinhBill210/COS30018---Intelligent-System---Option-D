#!/usr/bin/env python3
"""
Golden Test Dataset for Business Intelligence Agent Evaluation
============================================================

This module creates a comprehensive test dataset for evaluating the agent's:
1. Tool Selection Accuracy (Agent Reasoning)
2. Response Quality (Final Answer)

Structure: Each test case contains:
- query: Natural language question
- context: Business ID/name for repeatability  
- expected_tool_chain: Ordered list of tools that should be called
- expected_answer_summary: Key information that must be in good response
- category: Primary purpose category
"""

import json
from typing import List, Dict, Any
from datetime import datetime

class GoldenTestDataset:
    """Creates structured test cases for agent evaluation"""
    
    def __init__(self):
        self.test_cases = []
        
        # Real business data from the system
        self.sample_businesses = {
            "vietnamese_food_truck": {
                "business_id": "eEOYSgkmpB90uNA7lDOMRA", 
                "name": "Vietnamese Food Truck",
                "city": "Tampa Bay"
            },
            "st_honore_pastries": {
                "business_id": "MTSW4McQd7CbVtyjqoe9mw",
                "name": "St Honore Pastries", 
                "city": "Philadelphia"
            },
            "zios_italian": {
                "business_id": "0bPLkL0QhhPO5kt1_EXmNQ",
                "name": "Zio's Italian Market",
                "city": "Largo"
            },
            "tuna_bar": {
                "business_id": "MUTTqe8uqyMdBl186RmNeA",
                "name": "Tuna Bar",
                "city": "Philadelphia"
            },
            "roast_coffeehouse": {
                "business_id": "WKMJwqnfZKsAae75RMP6jA",
                "name": "Roast Coffeehouse and Wine Bar",
                "city": "Edmonton"
            },
            "sonic_nashville": {
                "business_id": "bBDDEgkFA1Otx9Lfe7BZUQ",
                "name": "Sonic Drive-In",
                "city": "Nashville"
            },
            "dennys": {
                "business_id": "il_Ro8jwPlHresjw9EGmBg",
                "name": "Denny's",
                "city": "Indianapolis"
            },
            "bap_philly": {
                "business_id": "ROeacJQwBeh05Rqg7F6TCg",
                "name": "BAP",
                "city": "Philadelphia"
            }
        }
    
    def create_hybrid_retrieve_queries(self) -> List[Dict[str, Any]]:
        """Create 10 HybridRetrieve test queries (evidence search)"""
        
        queries = [
            {
                "query": "Find three recent reviews that mention 'dirty tables' for Vietnamese Food Truck",
                "context": self.sample_businesses["vietnamese_food_truck"],
                "expected_tool_chain": ["business_fuzzy_search", "hybrid_retrieve"],
                "expected_answer_summary": [
                    "Returns specific reviews mentioning dirty tables",
                    "Includes review dates and star ratings", 
                    "Provides direct quotes as evidence",
                    "Filters results to be recent and relevant"
                ],
                "category": "Evidence Search"
            },
            {
                "query": "Show me customer complaints about food quality at St Honore Pastries with exact quotes",
                "context": self.sample_businesses["st_honore_pastries"],
                "expected_tool_chain": ["get_business_id", "hybrid_retrieve"],
                "expected_answer_summary": [
                    "Retrieves reviews specifically about food quality issues",
                    "Provides verbatim quotes from customer reviews",
                    "Includes metadata like review IDs and dates",
                    "Focuses on negative sentiment regarding food"
                ],
                "category": "Evidence Search"
            },
            {
                "query": "Find reviews from the last 6 months where customers praised the service at Zio's Italian Market",
                "context": self.sample_businesses["zios_italian"],
                "expected_tool_chain": ["get_business_id", "hybrid_retrieve"],
                "expected_answer_summary": [
                    "Filters reviews by date range (last 6 months)",
                    "Focuses on positive service mentions",
                    "Provides specific examples and quotes",
                    "Includes review timestamps for verification"
                ],
                "category": "Evidence Search"
            },
            {
                "query": "What specific evidence shows customers are unhappy with wait times at Tuna Bar?",
                "context": self.sample_businesses["tuna_bar"],
                "expected_tool_chain": ["get_business_id", "hybrid_retrieve"],
                "expected_answer_summary": [
                    "Searches for wait time related complaints",
                    "Provides direct customer quotes about delays",
                    "Includes star ratings from relevant reviews",
                    "Shows frequency/pattern of wait time issues"
                ],
                "category": "Evidence Search"
            },
            {
                "query": "Find reviews mentioning 'parking' issues at Roast Coffeehouse and Wine Bar",
                "context": self.sample_businesses["roast_coffeehouse"],
                "expected_tool_chain": ["business_fuzzy_search", "hybrid_retrieve"],
                "expected_answer_summary": [
                    "Retrieves reviews specifically about parking",
                    "Provides exact customer quotes about parking problems",
                    "Includes review context and ratings",
                    "Shows evidence of parking-related frustrations"
                ],
                "category": "Evidence Search"
            },
            {
                "query": "Show me positive reviews about the atmosphere at BAP with customer quotes",
                "context": self.sample_businesses["bap_philly"],
                "expected_tool_chain": ["get_business_id", "hybrid_retrieve"],
                "expected_answer_summary": [
                    "Searches for positive atmosphere/ambiance mentions",
                    "Provides direct quotes praising the environment",
                    "Includes high-rated reviews as evidence",
                    "Demonstrates positive customer experiences"
                ],
                "category": "Evidence Search"
            },
            {
                "query": "Find evidence of cleanliness complaints at Sonic Drive-In Nashville from customer reviews",
                "context": self.sample_businesses["sonic_nashville"],
                "expected_tool_chain": ["business_fuzzy_search", "hybrid_retrieve"],
                "expected_answer_summary": [
                    "Retrieves reviews mentioning cleanliness issues",
                    "Provides specific customer quotes about hygiene",
                    "Includes star ratings showing impact on satisfaction",
                    "Shows pattern of cleanliness concerns"
                ],
                "category": "Evidence Search"
            },
            {
                "query": "What do customers say about the coffee quality at Roast Coffeehouse? Give me exact quotes.",
                "context": self.sample_businesses["roast_coffeehouse"],
                "expected_tool_chain": ["get_business_id", "hybrid_retrieve"],
                "expected_answer_summary": [
                    "Focuses specifically on coffee quality mentions",
                    "Provides verbatim customer quotes about coffee",
                    "Includes both positive and negative feedback",
                    "Shows evidence with review metadata"
                ],
                "category": "Evidence Search"
            },
            {
                "query": "Find reviews where customers mention slow service at Denny's Indianapolis with specific examples",
                "context": self.sample_businesses["dennys"],
                "expected_tool_chain": ["business_fuzzy_search", "hybrid_retrieve"],
                "expected_answer_summary": [
                    "Searches for slow service related complaints",
                    "Provides specific customer examples and quotes",
                    "Includes review dates and star ratings",
                    "Shows evidence of service speed issues"
                ],
                "category": "Evidence Search"
            },
            {
                "query": "Show me customer reviews praising the value for money at Vietnamese Food Truck",
                "context": self.sample_businesses["vietnamese_food_truck"],
                "expected_tool_chain": ["get_business_id", "hybrid_retrieve"],
                "expected_answer_summary": [
                    "Retrieves reviews about value/pricing satisfaction",
                    "Provides quotes about good value or reasonable prices",
                    "Includes high-rated reviews as evidence",
                    "Demonstrates customer appreciation for pricing"
                ],
                "category": "Evidence Search"
            }
        ]
        
        return queries
    
    def create_business_pulse_queries(self) -> List[Dict[str, Any]]:
        """Create 10 BusinessPulse test queries (health snapshots)"""
        
        queries = [
            {
                "query": "Give me a quick overview of customer sentiment for Vietnamese Food Truck",
                "context": self.sample_businesses["vietnamese_food_truck"],
                "expected_tool_chain": ["get_business_id", "business_pulse"],
                "expected_answer_summary": [
                    "Provides overall sentiment distribution (positive/negative/neutral)",
                    "Shows star rating summary and review count",
                    "Identifies top positive and negative themes",
                    "Includes recent performance indicators"
                ],
                "category": "Health Snapshot"
            },
            {
                "query": "How is St Honore Pastries performing overall? Give me the health snapshot.",
                "context": self.sample_businesses["st_honore_pastries"],
                "expected_tool_chain": ["get_business_id", "business_pulse"],
                "expected_answer_summary": [
                    "Overall business health score or status",
                    "Key performance metrics (ratings, review volume)",
                    "Major strengths and weaknesses summary",
                    "Recent trends in customer satisfaction"
                ],
                "category": "Health Snapshot"
            },
            {
                "query": "What's the current customer satisfaction status at Zio's Italian Market?",
                "context": self.sample_businesses["zios_italian"],
                "expected_tool_chain": ["get_business_id", "business_pulse"],
                "expected_answer_summary": [
                    "Current satisfaction level/score",
                    "Distribution of customer ratings",
                    "Recent review trends (improving/declining)",
                    "Key satisfaction drivers identified"
                ],
                "category": "Health Snapshot"
            },
            {
                "query": "Show me the business health summary for Tuna Bar this quarter",
                "context": self.sample_businesses["tuna_bar"],
                "expected_tool_chain": ["get_business_id", "business_pulse"],
                "expected_answer_summary": [
                    "Quarterly performance summary",
                    "Review volume and rating trends",
                    "Top customer themes (positive and negative)",
                    "Consistency metrics and mismatch indicators"
                ],
                "category": "Health Snapshot"
            },
            {
                "query": "What's the overall reputation status of Roast Coffeehouse and Wine Bar?",
                "context": self.sample_businesses["roast_coffeehouse"],
                "expected_tool_chain": ["business_fuzzy_search", "business_pulse"],
                "expected_answer_summary": [
                    "Overall reputation score/status",
                    "Key reputation drivers (strengths/weaknesses)",
                    "Customer sentiment distribution",
                    "Reputation trends over time"
                ],
                "category": "Health Snapshot"
            },
            {
                "query": "Give me a high-level performance overview for BAP restaurant",
                "context": self.sample_businesses["bap_philly"],
                "expected_tool_chain": ["get_business_id", "business_pulse"],
                "expected_answer_summary": [
                    "High-level performance indicators",
                    "Overall customer satisfaction metrics",
                    "Key business strengths and challenges",
                    "Performance trends and patterns"
                ],
                "category": "Health Snapshot"
            },
            {
                "query": "How healthy is the customer feedback situation at Sonic Drive-In Nashville?",
                "context": self.sample_businesses["sonic_nashville"],
                "expected_tool_chain": ["business_fuzzy_search", "business_pulse"],
                "expected_answer_summary": [
                    "Feedback health assessment (volume, sentiment)",
                    "Rating distribution and trends",
                    "Main customer themes and concerns",
                    "Overall feedback quality indicators"
                ],
                "category": "Health Snapshot"
            },
            {
                "query": "What's the customer experience health check for Denny's Indianapolis?",
                "context": self.sample_businesses["dennys"],
                "expected_tool_chain": ["business_fuzzy_search", "business_pulse"],
                "expected_answer_summary": [
                    "Customer experience health status",
                    "Experience quality metrics and scores", 
                    "Top experience drivers (positive and negative)",
                    "Experience trend analysis"
                ],
                "category": "Health Snapshot"
            },
            {  
                "query": "Show me the current business performance dashboard for St Honore Pastries",
                "context": self.sample_businesses["st_honore_pastries"],
                "expected_tool_chain": ["get_business_id", "business_pulse"],
                "expected_answer_summary": [
                    "Dashboard-style performance overview",
                    "Key performance metrics and KPIs",
                    "Performance trend indicators",
                    "Critical areas requiring attention"
                ],
                "category": "Health Snapshot"
            },
            {
                "query": "What's the overall customer sentiment pulse for Vietnamese Food Truck recently?",
                "context": self.sample_businesses["vietnamese_food_truck"],
                "expected_tool_chain": ["get_business_id", "business_pulse"],
                "expected_answer_summary": [
                    "Recent sentiment pulse and trends",
                    "Sentiment distribution changes over time",
                    "Recent positive and negative themes",
                    "Sentiment momentum indicators"
                ],
                "category": "Health Snapshot"
            }
        ]
        
        return queries
    
    def create_aspect_absa_queries(self) -> List[Dict[str, Any]]:
        """Create 8 AspectABSA test queries (sentiment analysis)"""
        
        queries = [
            {
                "query": "What specific aspects are customers happiest about at Vietnamese Food Truck?",
                "context": self.sample_businesses["vietnamese_food_truck"],
                "expected_tool_chain": ["get_business_id", "analyze_aspects"],
                "expected_answer_summary": [
                    "Identifies top positive aspects (food, service, value, etc.)",
                    "Provides sentiment scores for each aspect",
                    "Shows support/mention frequency for each aspect",
                    "Includes example quotes for positive aspects"
                ],
                "category": "Aspect Sentiment Analysis"
            },
            {
                "query": "Break down customer sentiment by specific areas for St Honore Pastries",
                "context": self.sample_businesses["st_honore_pastries"],
                "expected_tool_chain": ["get_business_id", "analyze_aspects"],
                "expected_answer_summary": [
                    "Comprehensive aspect-level sentiment breakdown",
                    "Sentiment scores for food, service, ambiance, price",
                    "Support metrics showing which aspects get mentioned most",
                    "Evidence quotes for each major aspect"
                ],
                "category": "Aspect Sentiment Analysis"
            },
            {
                "query": "Which aspects of Zio's Italian Market are customers most critical about?",
                "context": self.sample_businesses["zios_italian"],
                "expected_tool_chain": ["get_business_id", "analyze_aspects"],
                "expected_answer_summary": [
                    "Identifies aspects with most negative sentiment",
                    "Ranks aspects by severity of criticism",
                    "Shows negative sentiment scores and patterns",
                    "Provides critical customer quotes for problem aspects"
                ],
                "category": "Aspect Sentiment Analysis"
            },
            {
                "query": "Analyze the different aspects customers mention about Tuna Bar - what are they saying?",
                "context": self.sample_businesses["tuna_bar"],
                "expected_tool_chain": ["get_business_id", "analyze_aspects"],
                "expected_answer_summary": [
                    "Comprehensive aspect discovery and analysis",
                    "Sentiment analysis for each discovered aspect",
                    "Aspect mention frequency and importance",
                    "Representative quotes for each aspect"
                ],
                "category": "Aspect Sentiment Analysis"
            },
            {
                "query": "What are the sentiment patterns for different aspects at Roast Coffeehouse?",
                "context": self.sample_businesses["roast_coffeehouse"],
                "expected_tool_chain": ["business_fuzzy_search", "analyze_aspects"],
                "expected_answer_summary": [
                    "Sentiment pattern analysis across aspects",
                    "Identifies consistent vs. inconsistent aspects",
                    "Shows aspect sentiment trends and variability",
                    "Highlights aspects with mixed sentiment"
                ],
                "category": "Aspect Sentiment Analysis"
            },
            {
                "query": "Break down what customers like and dislike about BAP restaurant by category",
                "context": self.sample_businesses["bap_philly"],
                "expected_tool_chain": ["get_business_id", "analyze_aspects"],
                "expected_answer_summary": [
                    "Clear categorization of likes vs. dislikes",
                    "Aspect-level sentiment comparison",
                    "Strength and weakness identification by aspect",
                    "Supporting evidence for positive and negative aspects"
                ],
                "category": "Aspect Sentiment Analysis"
            },
            {
                "query": "What specific service aspects are problematic at Sonic Drive-In Nashville?",
                "context": self.sample_businesses["sonic_nashville"],
                "expected_tool_chain": ["business_fuzzy_search", "analyze_aspects"],
                "expected_answer_summary": [
                    "Focus on service-related aspects specifically",
                    "Identifies problematic service elements",
                    "Shows severity of service aspect issues",
                    "Provides customer quotes about service problems"
                ],
                "category": "Aspect Sentiment Analysis"
            },
            {
                "query": "How do customers feel about different aspects of their experience at Denny's?",
                "context": self.sample_businesses["dennys"],
                "expected_tool_chain": ["business_fuzzy_search", "analyze_aspects"],
                "expected_answer_summary": [
                    "Comprehensive experience aspect analysis",
                    "Sentiment scores for all experience elements",
                    "Customer satisfaction patterns by aspect",
                    "Overall experience sentiment summary"
                ],
                "category": "Aspect Sentiment Analysis"
            }
        ]
        
        return queries
    
    def create_drift_segment_queries(self) -> List[Dict[str, Any]]:
        """Create 8 DriftSegment test queries (trend analysis)"""
        
        queries = [
            {
                "query": "Have complaints about wait times at Vietnamese Food Truck increased over the last six months?",
                "context": self.sample_businesses["vietnamese_food_truck"],
                "expected_tool_chain": ["get_business_id", "analyze_aspects", "business_pulse"],
                "expected_answer_summary": [
                    "Shows trend analysis for wait time complaints over 6 months",
                    "Compares current vs. previous period complaint frequency",
                    "Identifies any anomalies or spikes in wait time issues",
                    "Provides evidence from specific time periods"
                ],
                "category": "Trend Analysis"
            },
            {
                "query": "Are customer satisfaction scores for St Honore Pastries trending up or down this year?",
                "context": self.sample_businesses["st_honore_pastries"],
                "expected_tool_chain": ["get_business_id", "business_pulse"],
                "expected_answer_summary": [
                    "Shows satisfaction trend direction over the year",
                    "Identifies periods of improvement or decline",
                    "Compares different time windows for trend analysis",
                    "Highlights significant trend changes or patterns"
                ],
                "category": "Trend Analysis"
            },
            {
                "query": "How has the frequency of food quality complaints changed at Zio's Italian Market recently?",
                "context": self.sample_businesses["zios_italian"],
                "expected_tool_chain": ["get_business_id", "analyze_aspects", "business_pulse"],
                "expected_answer_summary": [
                    "Tracks food quality complaint frequency over time",
                    "Shows recent changes in complaint patterns",
                    "Identifies any increases or decreases in issues",
                    "Provides temporal context for quality trends"
                ],
                "category": "Trend Analysis"
            },
            {
                "query": "What's the trend in customer ratings for Tuna Bar over the past year?",
                "context": self.sample_businesses["tuna_bar"],
                "expected_tool_chain": ["get_business_id", "business_pulse"],
                "expected_answer_summary": [
                    "Shows rating trends across the past year",
                    "Identifies patterns in rating changes",
                    "Highlights periods of high or low performance",
                    "Provides evidence of rating trajectory"
                ],
                "category": "Trend Analysis"
            },
            {
                "query": "Are service-related complaints at Roast Coffeehouse becoming more or less frequent?",
                "context": self.sample_businesses["roast_coffeehouse"],
                "expected_tool_chain": ["business_fuzzy_search", "analyze_aspects", "business_pulse"],
                "expected_answer_summary": [
                    "Analyzes service complaint frequency trends",
                    "Shows directional change in service issues",
                    "Compares different time periods for frequency analysis",
                    "Identifies any concerning increases or positive decreases"
                ],
                "category": "Trend Analysis"
            },
            {
                "query": "How do different customer segments rate BAP restaurant compared to overall averages?",
                "context": self.sample_businesses["bap_philly"],
                "expected_tool_chain": ["get_business_id", "business_pulse", "analyze_aspects"],
                "expected_answer_summary": [
                    "Segments customers by rating patterns or helpfulness",
                    "Compares segment performance to overall averages",
                    "Identifies which segments are most/least satisfied",
                    "Shows segment-specific trends and patterns"
                ],
                "category": "Trend Analysis"
            },
            {
                "query": "Is there a seasonal pattern in customer complaints at Sonic Drive-In Nashville?",
                "context": self.sample_businesses["sonic_nashville"],
                "expected_tool_chain": ["business_fuzzy_search", "business_pulse", "analyze_aspects"],
                "expected_answer_summary": [
                    "Identifies seasonal patterns in complaint frequency",
                    "Shows time-based variations in customer issues",
                    "Highlights peak complaint periods or seasons",
                    "Provides evidence of cyclical complaint patterns"
                ],
                "category": "Trend Analysis"
            },
            {
                "query": "Have positive reviews about atmosphere increased at Denny's Indianapolis recently?",
                "context": self.sample_businesses["dennys"],
                "expected_tool_chain": ["business_fuzzy_search", "analyze_aspects", "business_pulse"],
                "expected_answer_summary": [
                    "Tracks positive atmosphere mentions over time",
                    "Shows recent changes in atmosphere satisfaction",
                    "Compares current vs. previous period positive feedback",
                    "Identifies trends in specific aspect improvement"
                ],
                "category": "Trend Analysis"
            }
        ]
        
        return queries
    
    def create_impact_ranker_queries(self) -> List[Dict[str, Any]]:
        """Create 7 ImpactRanker test queries (priority scoring)"""
        
        queries = [
            {
                "query": "Of all the negative feedback for Vietnamese Food Truck, what is the single most impactful problem?",
                "context": self.sample_businesses["vietnamese_food_truck"],
                "expected_tool_chain": ["get_business_id", "analyze_aspects", "business_pulse"],
                "expected_answer_summary": [
                    "Identifies the highest impact negative issue",
                    "Provides impact scoring methodology (severity × frequency × recency)",
                    "Explains why this issue ranks highest",
                    "Shows comparative impact scores for other issues"
                ],
                "category": "Priority Scoring"
            },
            {
                "query": "Rank the top 3 areas St Honore Pastries should prioritize for improvement",
                "context": self.sample_businesses["st_honore_pastries"],
                "expected_tool_chain": ["get_business_id", "analyze_aspects", "business_pulse"],
                "expected_answer_summary": [
                    "Provides prioritized ranking of improvement areas",
                    "Shows impact scores for each prioritized area",
                    "Explains ranking methodology and criteria",
                    "Includes supporting evidence for each priority"
                ],
                "category": "Priority Scoring"
            },
            {
                "query": "What's the most urgent customer experience issue at Zio's Italian Market based on impact?",
                "context": self.sample_businesses["zios_italian"],
                "expected_tool_chain": ["get_business_id", "analyze_aspects", "business_pulse"],
                "expected_answer_summary": [
                    "Identifies the most urgent experience issue",
                    "Provides urgency/impact scoring rationale",
                    "Shows why this issue requires immediate attention",
                    "Compares urgency levels of different issues"
                ],
                "category": "Priority Scoring"
            },
            {
                "query": "Which aspect of Tuna Bar has the highest potential impact if improved?",
                "context": self.sample_businesses["tuna_bar"],
                "expected_tool_chain": ["get_business_id", "analyze_aspects", "business_pulse"],
                "expected_answer_summary": [
                    "Identifies aspect with highest improvement potential",
                    "Shows potential impact calculation and reasoning",
                    "Explains why improving this aspect would be most beneficial",
                    "Provides comparative potential impact for other aspects"
                ],
                "category": "Priority Scoring"
            },
            {
                "query": "Score and rank the business issues at Roast Coffeehouse by severity and frequency",
                "context": self.sample_businesses["roast_coffeehouse"],
                "expected_tool_chain": ["business_fuzzy_search", "analyze_aspects", "business_pulse"],
                "expected_answer_summary": [
                    "Provides comprehensive issue scoring and ranking",
                    "Shows severity and frequency metrics for each issue",
                    "Explains scoring methodology and weightings",
                    "Presents ranked list with supporting scores"
                ],
                "category": "Priority Scoring"
            },
            {
                "query": "What's the most cost-effective improvement BAP restaurant could make based on customer feedback impact?",
                "context": self.sample_businesses["bap_philly"],
                "expected_tool_chain": ["get_business_id", "analyze_aspects", "business_pulse"],
                "expected_answer_summary": [
                    "Identifies improvement with best cost-effectiveness ratio",
                    "Considers both customer impact and implementation ease",
                    "Explains cost-effectiveness calculation",
                    "Compares multiple improvement options for ROI"
                ],
                "category": "Priority Scoring"
            },
            {
                "query": "Prioritize the customer complaints at Sonic Drive-In Nashville by business impact potential",
                "context": self.sample_businesses["sonic_nashville"],
                "expected_tool_chain": ["business_fuzzy_search", "analyze_aspects", "business_pulse"],
                "expected_answer_summary": [
                    "Prioritizes complaints by potential business impact",
                    "Shows impact scoring for different complaint types",
                    "Explains business impact assessment criteria",
                    "Provides prioritized action list based on impact"
                ],
                "category": "Priority Scoring"
            }
        ]
        
        return queries
    
    def create_action_planner_queries(self) -> List[Dict[str, Any]]:
        """Create 7 ActionPlanner test queries (recommendations)"""
        
        queries = [
            {
                "query": "Suggest two low-cost improvements I can make to address feedback about 'slow service' at Vietnamese Food Truck",
                "context": self.sample_businesses["vietnamese_food_truck"],
                "expected_tool_chain": ["get_business_id", "analyze_aspects", "create_action_plan"],
                "expected_answer_summary": [
                    "Provides 2 specific low-cost improvement recommendations",
                    "Addresses slow service issues specifically",
                    "Includes implementation steps and resource requirements",
                    "Estimates effort/impact and provides KPIs to track"
                ],
                "category": "Action Recommendations"
            },
            {
                "query": "Create an action plan to improve customer satisfaction at St Honore Pastries within 8 weeks",
                "context": self.sample_businesses["st_honore_pastries"],
                "expected_tool_chain": ["get_business_id", "analyze_aspects", "create_action_plan"],
                "expected_answer_summary": [
                    "Provides comprehensive 8-week action plan",
                    "Includes specific actions with timelines",
                    "Shows expected outcomes and success metrics",
                    "Considers resource constraints and feasibility"
                ],
                "category": "Action Recommendations"
            },
            {
                "query": "What specific actions should Zio's Italian Market take to address their top customer complaints?",
                "context": self.sample_businesses["zios_italian"],
                "expected_tool_chain": ["get_business_id", "analyze_aspects", "create_action_plan"],
                "expected_answer_summary": [
                    "Provides targeted actions for top complaints",
                    "Links each action to specific customer feedback",
                    "Includes implementation guidance and effort estimates",
                    "Shows expected impact on customer satisfaction"
                ],
                "category": "Action Recommendations"
            },
            {
                "query": "Design a customer experience improvement plan for Tuna Bar with a medium budget",
                "context": self.sample_businesses["tuna_bar"],
                "expected_tool_chain": ["get_business_id", "analyze_aspects", "create_action_plan"],
                "expected_answer_summary": [
                    "Provides experience improvement plan within medium budget",
                    "Balances impact with budget constraints",
                    "Includes specific customer experience enhancements",
                    "Shows ROI projections and success metrics"
                ],
                "category": "Action Recommendations"
            },
            {
                "query": "What operational changes should Roast Coffeehouse implement to improve efficiency based on reviews?",
                "context": self.sample_businesses["roast_coffeehouse"],
                "expected_tool_chain": ["business_fuzzy_search", "analyze_aspects", "create_action_plan"],
                "expected_answer_summary": [
                    "Focuses on operational efficiency improvements",
                    "Links recommendations to customer feedback about efficiency",
                    "Provides implementation steps and resource needs",
                    "Includes efficiency metrics and tracking KPIs"
                ],
                "category": "Action Recommendations"
            },
            {
                "query": "Generate a staff training action plan for BAP restaurant to address service quality issues",
                "context": self.sample_businesses["bap_philly"],
                "expected_tool_chain": ["get_business_id", "analyze_aspects", "create_action_plan"],
                "expected_answer_summary": [
                    "Provides comprehensive staff training plan",
                    "Addresses specific service quality issues from reviews",
                    "Includes training modules, timeline, and resources",
                    "Shows expected service quality improvements"
                ],
                "category": "Action Recommendations"
            },
            {
                "query": "What quick wins can Sonic Drive-In Nashville implement this month to boost ratings?",
                "context": self.sample_businesses["sonic_nashville"],
                "expected_tool_chain": ["business_fuzzy_search", "analyze_aspects", "create_action_plan"],
                "expected_answer_summary": [
                    "Identifies quick, implementable improvements",
                    "Focuses on actions that can boost ratings rapidly",
                    "Provides 1-month implementation timeline",
                    "Shows expected rating impact and tracking metrics"
                ],
                "category": "Action Recommendations"
            }
        ]
        
        return queries

    def generate_all_test_cases(self) -> List[Dict[str, Any]]:
        """Generate all test cases for the golden dataset"""
        
        # Generate all query categories
        all_queries = []
        all_queries.extend(self.create_hybrid_retrieve_queries())
        all_queries.extend(self.create_business_pulse_queries())
        all_queries.extend(self.create_aspect_absa_queries())
        all_queries.extend(self.create_drift_segment_queries())
        all_queries.extend(self.create_impact_ranker_queries())
        all_queries.extend(self.create_action_planner_queries())
        
        # Add metadata to each query
        for i, query in enumerate(all_queries):
            query["test_id"] = f"test_{i+1:03d}"
            query["created_at"] = datetime.now().isoformat()
            
        return all_queries
    
    def save_dataset(self, filepath: str = "evaluation/golden_test_dataset.json"):
        """Save the complete golden test dataset"""
        
        test_cases = self.generate_all_test_cases()
        
        dataset = {
            "metadata": {
                "name": "Business Intelligence Agent Golden Test Dataset",
                "version": "1.0",
                "created_at": datetime.now().isoformat(),
                "total_test_cases": len(test_cases),
                "tools_covered": ["HybridRetrieve", "BusinessPulse", "AspectABSA", "DriftSegment", "ImpactRanker", "ActionPlanner"],
                "description": "Comprehensive test dataset for evaluating agent tool selection and response quality"
            },
            "test_cases": test_cases,
            "statistics": {
                "hybrid_retrieve": len([t for t in test_cases if t["category"] == "Evidence Search"]),
                "business_pulse": len([t for t in test_cases if t["category"] == "Health Snapshot"]),
                "aspect_absa": len([t for t in test_cases if t["category"] == "Aspect Sentiment Analysis"]),
                "drift_segment": len([t for t in test_cases if t["category"] == "Trend Analysis"]),
                "impact_ranker": len([t for t in test_cases if t["category"] == "Priority Scoring"]),
                "action_planner": len([t for t in test_cases if t["category"] == "Action Recommendations"])
            }
        }
        
        # Create directory if it doesn't exist
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(dataset, f, indent=2)
            
        return dataset, filepath

if __name__ == "__main__":
    # Create and save the dataset
    creator = GoldenTestDataset()
    dataset, filepath = creator.save_dataset()
    
    print(f"✅ Golden Test Dataset Created!")
    print(f"📁 Saved to: {filepath}")
    print(f"📊 Total test cases: {dataset['metadata']['total_test_cases']}")
    print(f"🔍 HybridRetrieve queries: {dataset['statistics']['hybrid_retrieve']}")
    print(f"💓 BusinessPulse queries: {dataset['statistics']['business_pulse']}")
    print(f"🎯 AspectABSA queries: {dataset['statistics']['aspect_absa']}")
    print(f"📊 DriftSegment queries: {dataset['statistics']['drift_segment']}")
    print(f"🎖️ ImpactRanker queries: {dataset['statistics']['impact_ranker']}")
    print(f"🎯 ActionPlanner queries: {dataset['statistics']['action_planner']}")
    print(f"\n🎉 Complete Golden Test Dataset Ready for Evaluation!")
