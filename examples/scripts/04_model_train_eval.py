import pandas as pd
from matkit.ml import train_model, evaluate_model

if __name__ == "__main__":
    X_train = pd.read_csv("X_train.csv")
    X_test = pd.read_csv("X_test.csv")
    y_train = pd.read_csv("y_train.csv").values.ravel()
    y_test = pd.read_csv("y_test.csv").values.ravel()

    model = train_model(X_train, y_train, model_type='RandomForest')
    results = evaluate_model(model, X_test, y_test)

    print("Evaluation Metrics:")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")
