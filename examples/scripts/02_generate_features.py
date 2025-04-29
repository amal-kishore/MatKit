import pandas as pd
from matkit.featurize import generate_features

if __name__ == "__main__":
    df = pd.read_csv("abc_clean_data.csv")
    df_feat = generate_features(df, featureset='composition', save=True, filename="abc_features.csv")
    print(f"Feature matrix shape: {df_feat.shape}")
