import pandas as pd
import matplotlib.pyplot as plt
from matkit.ml import get_feature_importance

if __name__ == "__main__":
    X_train = pd.read_csv("X_train.csv")
    y_train = pd.read_csv("y_train.csv").values.ravel()

    model, importances = get_feature_importance(X_train, y_train, model_type='RandomForest')

    imp_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': importances})
    imp_df.sort_values(by="Importance", ascending=False).to_csv("feature_importances.csv", index=False)

    imp_df.nlargest(20, "Importance").plot.bar(x='Feature', y='Importance', figsize=(10,5), title="Top 20 Features")
    plt.tight_layout()
    plt.savefig("feature_importance_plot.png")
