import argparse
import mlflow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, learning_curve
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Parse argumen dari MLproject
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True)
args = parser.parse_args()
dataset_path = args.data_path

# Setup MLflow
mlflow.autolog()

# Load data
df = pd.read_csv(dataset_path)
X = df.drop("Sleep Quality", axis=1)
y = df["Sleep Quality"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run():
    rf = RandomForestRegressor(
        n_estimators=200, max_depth=10, min_samples_split=5,
        min_samples_leaf=2, max_features='sqrt', random_state=42,
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    print("MAE :", mean_absolute_error(y_test, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
    print("R²  :", r2_score(y_test, y_pred))

    # Plot dan log prediksi vs aktual
    plt.figure(figsize=(5, 5))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    plt.xlabel('True Sleep Quality')
    plt.ylabel('Predicted')
    plt.title('Prediksi vs Aktual')
    plt.tight_layout()
    plt.savefig("prediksi_vs_aktual.png")
    mlflow.log_artifact("prediksi_vs_aktual.png")

    # Plot dan log learning curve
    train_sizes, train_scores, test_scores = learning_curve(
        rf, X, y, cv=5, scoring='neg_mean_squared_error',
        train_sizes=np.linspace(0.1, 1.0, 5), n_jobs=-1
    )
    train_rmse = np.sqrt(-train_scores)
    test_rmse = np.sqrt(-test_scores)

    plt.figure(figsize=(7, 5))
    plt.plot(train_sizes, train_rmse.mean(axis=1), 'o-', label="Training RMSE")
    plt.plot(train_sizes, test_rmse.mean(axis=1), 'o-', label="Validation RMSE")
    plt.title("Learning Curve – Random Forest")
    plt.xlabel("Train Size")
    plt.ylabel("RMSE")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("learning_curve_rf.png")
    mlflow.log_artifact("learning_curve_rf.png")

    mlflow.log_artifact(dataset_path, artifact_path="datasets")
