import pandas as pd
from pathlib import Path
import tempfile
from tools.aspect_analysis import AspectABSAToolParquet

def test_aspect_absa_tool_parquet(tmp_path):
    print("Testing AspectABSAToolParquet with a sample parquet file...")

    # Create a tiny DataFrame with a few reviews
    df = pd.DataFrame({
        "text": [
            "The pizza was amazing but delivery was late",
            "Service was friendly and quick",
            "Overpriced for the portion size"
        ]
    })

    # Save to a temporary parquet file
    parquet_file = tmp_path / "sample_reviews.parquet"
    df.to_parquet(parquet_file, engine="pyarrow")

    tool = AspectABSAToolParquet()
    result = tool(str(parquet_file))

    print(f"Output: {result}")
    assert result is not None and "aspects" in result

if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        test_aspect_absa_tool_parquet(tmp_path=Path(tmp))
