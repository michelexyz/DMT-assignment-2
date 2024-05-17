import pandas as pd
import math

def missingness_and_uniqueness(df):
    stats = pd.DataFrame(index = ["missing_values", "count", "n_unique","uniqueness_factor", "percentage_missing"], columns=df.columns)

    for c in df.columns:
        stats.loc["missing_values", c] = df[c].isna().sum()
        stats.loc["count", c] = df[c].count()
        stats.loc["n_unique", c] = df[c].nunique()
        stats.loc["uniqueness_factor", c] = -math.log(df[c].nunique() / stats.loc["count", c]) 
        stats.loc["percentage_missing", c] = (df[c].isna().sum() / df.shape[0] * 100).round(2)

    stats.loc["uniqueness_factor"] = stats.loc["uniqueness_factor"].astype(float).round(2)

    return stats


