import pandas as pd
from matkit.utils import filter_dataframe

if __name__ == "__main__":
    df = pd.read_csv("abc_raw_data.csv")
    df_clean = filter_dataframe(df, min_elements=3, max_elements=3, remove_duplicates=True)
    df_clean.to_csv("abc_clean_data.csv", index=False)
    print(f"Filtered data to {len(df_clean)} entries.")
