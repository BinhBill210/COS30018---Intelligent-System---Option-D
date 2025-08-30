# test/test_aspect_analysis.py
from tools.aspect_analysis1 import AspectABSATool


def test_aspect_absa_tool():
    print("Testing AspectABSATool...")
    tool = AspectABSATool("./chroma_db")

    # Test with a business_id you know exists in your ChromaDB
    business_id = "XQfwVwDr-v0ZS3_CbbE5Xw"
    result = tool(business_id=business_id)

    print(f"Business ID: {business_id}")
    print("Output:")
    print(result)

    # Simple check
    return result is not None and "aspects" in result

if __name__ == "__main__":
    test_aspect_absa_tool()
