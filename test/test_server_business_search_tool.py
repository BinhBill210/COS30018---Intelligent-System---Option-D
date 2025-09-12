from tools.server_business_search_tool import BusinessSearchTool

def test_business_search_tool():
    print("Testing BusinessSearchTool...")
    tool = BusinessSearchTool()

    # Test exact name lookup
    name = "Vietnamese Food Truck"
    business_id = tool.get_business_id(name)
    print(f"Business ID for '{name}': {business_id}")
    assert business_id is not None

    # Test fuzzy search
    fuzzy_results = tool.fuzzy_search("Vietnamese Food Truck", top_n=3)
    print("Fuzzy search results for 'Vietnamese Food Truck':")
    for res in fuzzy_results:
        print(res)
    assert len(fuzzy_results) > 0

    # Test semantic search
    semantic_results = tool.search_businesses("Vietnamese Food Truck")
    print("Semantic search results for 'Vietnamese Food Truck':")
    for res in semantic_results:
        print(res)
    assert len(semantic_results) > 0

    # Test general info
    if business_id:
        info = tool.get_business_info(business_id)
        print(f"General info for business_id '{business_id}':")
        print(info)
        assert info is not None and 'name' in info

    print("BusinessSearchTool test passed!")
    return True

if __name__ == "__main__":
    test_business_search_tool()
