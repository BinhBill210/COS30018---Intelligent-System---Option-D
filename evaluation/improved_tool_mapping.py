
def _normalize_tool_name(self, tool_name: str) -> str:
    """Enhanced tool name normalization"""
    mapping = {
        # Search and retrieval tools
        'search_reviews': 'hybrid_retrieve',
        'hybrid_retrieve': 'hybrid_retrieve',
        'fuzzy_search': 'business_search',
        'business_fuzzy_search': 'business_search',
        'search_businesses': 'business_search',
        'get_business_id': 'business_search',
        'get_business_info': 'business_search',
                
        # Analysis tools
        'analyze_sentiment': 'aspect_analysis',
        'analyze_aspects': 'aspect_analysis',
        'aspect_analysis': 'aspect_analysis',
                
        # Business intelligence tools
        'get_data_summary': 'business_pulse',
        'business_pulse': 'business_pulse',
        'data_summary': 'business_pulse',
                
         # Action and planning tools
        'create_action_plan': 'action_planner',
        'action_planner': 'action_planner',
        'generate_review_response': 'review_response',
        'review_response': 'review_response'
    }
            
    normalized = tool_name.lower().strip()
    return mapping.get(normalized, normalized)
        