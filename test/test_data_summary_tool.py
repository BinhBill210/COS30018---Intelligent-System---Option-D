from dotenv import load_dotenv
load_dotenv()
from tools.data_summary_tool import DataSummaryTool

def test_data_summary_tool():
    print("Testing DataSummaryTool...")
    data_tool = DataSummaryTool("./data/processed/review_cleaned.parquet")
    result = data_tool("XQfwVwDr-v0ZS3_CbbE5Xw")
    print(f"Output: {result}")
    return result is not None and 'total_reviews' in result and result['total_reviews'] > 0

if __name__ == "__main__":
    test_data_summary_tool()
