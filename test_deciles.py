import polars as pl
import numpy as np
from array_utils import assign_deciles_to_dataframe

# Example usage
if __name__ == "__main__":
    # Create a sample dataframe
    df = pl.DataFrame({
        "id": range(1, 21),
        "scores": [12, 25, 8, 45, 67, 23, 89, 34, 56, 78, 
                  91, 15, 38, 72, 29, 84, 41, 63, 17, 95]
    })
    
    # Create a reference series to compute deciles from
    # This could be a different dataset, historical data, etc.
    reference_data = np.random.normal(50, 20, 1000)  # Normal distribution with mean=50, std=20
    
    print("Original DataFrame:")
    print(df)
    print()
    
    # Assign deciles based on the reference data
    df_with_deciles = assign_deciles_to_dataframe(df, reference_data, "scores")
    
    print("DataFrame with deciles:")
    print(df_with_deciles)
    print()
    
    # Show distribution of deciles
    decile_counts = df_with_deciles.group_by("decile").agg(pl.len().alias("count")).sort("decile")
    print("Decile distribution:")
    print(decile_counts)
    print()
    
    # Show some statistics
    decile_stats = df_with_deciles.group_by("decile").agg([
        pl.col("scores").min().alias("min_score"),
        pl.col("scores").max().alias("max_score"),
        pl.col("scores").mean().alias("avg_score")
    ]).sort("decile")
    
    print("Decile statistics:")
    print(decile_stats) 