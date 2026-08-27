# # import pandas as pd
# # import joblib
# # from sklearn.model_selection import train_test_split
# # from sklearn.ensemble import RandomForestClassifier

# # # Load data
# # df = pd.read_csv("data.csv")  # save previous df to csv first

# # X = df[['cpu_usage']]
# # y = df['failure']

# # # Split data
# # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# # # Train model
# # model = RandomForestClassifier()
# # model.fit(X_train, y_train)

# # # Test accuracy
# # accuracy = model.score(X_test, y_test)
# # print("Accuracy:", accuracy)
# # joblib.dump(model, "model.pkl")


# import joblib
# import pandas as pd
# from sklearn.ensemble import IsolationForest
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# df = pd.read_csv("data.csv")

# if "cpu" not in df.columns and "cpu_usage" in df.columns:
# 	df = df.rename(columns={"cpu_usage": "cpu"})

# required_columns = ["cpu", "memory", "restarts"]
# missing_columns = [column for column in required_columns if column not in df.columns]

# if missing_columns:
# 	raise ValueError(f"Missing required columns in data.csv: {missing_columns}")

# X = df[required_columns]

# # If labels exist, do a train/test split and evaluate on test set
# if "failure" in df.columns:
# 	y = df["failure"]
# 	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if len(y.unique())>1 else None)

# 	model = IsolationForest(contamination=0.2, random_state=42, n_estimators=200)
# 	model.fit(X_train)

# 	preds = model.predict(X_test)
# 	y_pred = [1 if p == -1 else 0 for p in preds]

# 	print("Evaluation on test set:")
# 	print("Accuracy:", accuracy_score(y_test, y_pred))
# 	print("Precision (pos=1):", precision_score(y_test, y_pred, zero_division=0))
# 	print("Recall (pos=1):", recall_score(y_test, y_pred, zero_division=0))
# 	print("F1 (pos=1):", f1_score(y_test, y_pred, zero_division=0))
# 	print("\nClassification report:\n")
# 	print(classification_report(y_test, y_pred, zero_division=0))

# 	# Train on full training+test (optionally) or save the model trained on train
# 	joblib.dump(model, "model.pkl")
# 	print("Model trained on train split and saved to model.pkl")
# else:
# 	# No labels: train on full dataset (unsupervised)
# 	model = IsolationForest(contamination=0.2, random_state=42, n_estimators=200)
# 	model.fit(X)
# 	joblib.dump(model, "model.pkl")
# 	print("No labels found. Model trained on full dataset and saved to model.pkl")


import joblib
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# -----------------------------------
# LOAD DATA
# -----------------------------------

df = pd.read_csv("data.csv")

print(f"\nLoaded dataset: {len(df)} rows")

# -----------------------------------
# FEATURES USED FOR TRAINING
# -----------------------------------

FEATURE_COLUMNS = [
    "cpu_usage",
    "memory_working_set",
    "restart_count",
    "oom_events",
    "cpu_pressure",
    "io_pressure",
]

# -----------------------------------
# VALIDATE DATA
# -----------------------------------

missing_columns = [column for column in FEATURE_COLUMNS if column not in df.columns]

if missing_columns:

    raise ValueError(
        f"\nMissing required columns " f"in data.csv: " f"{missing_columns}"
    )

# -----------------------------------
# FEATURE MATRIX
# -----------------------------------

# X = df[FEATURE_COLUMNS]

# Keep only ML numeric features

X = df[
    [
        "cpu_usage",
        "memory_working_set",
        "restart_count",
        "oom_events",
        "cpu_pressure",
        "io_pressure",
    ]
]

# Convert safely to numeric

X = X.apply(pd.to_numeric, errors="coerce")

# Replace NaN with 0

X = X.fillna(0)
# -----------------------------------
# TRAIN WITH LABELS
# -----------------------------------

if "failure" in df.columns:

    y = df["failure"]

    print(f"\nFailure label distribution:\n")

    print(y.value_counts())

    # -----------------------------------
    # TRAIN TEST SPLIT
    # -----------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=(y if len(y.unique()) > 1 else None),
    )

    # -----------------------------------
    # MODEL
    # -----------------------------------

    # model = IsolationForest(

    #     contamination=0.2,

    #     random_state=42,

    #     n_estimators=200,
    # )

    model = IsolationForest(
        n_estimators=300,
        max_samples="auto",
        contamination=0.08,
        max_features=0.9,
        bootstrap=False,
        random_state=42,
    )

    # -----------------------------------
    # TRAIN
    # -----------------------------------

    model.fit(X_train)

    # -----------------------------------
    # PREDICT
    # -----------------------------------

    predictions = model.predict(X_test)

    # IsolationForest:
    # -1 = anomaly
    #  1 = normal

    y_pred = [1 if prediction == -1 else 0 for prediction in predictions]

    # -----------------------------------
    # METRICS
    # -----------------------------------

    print("\nEvaluation Results:\n")

    print("Accuracy:", accuracy_score(y_test, y_pred) * 100)

    print("Precision:", precision_score(y_test, y_pred, zero_division=0))

    print("Recall:", recall_score(y_test, y_pred, zero_division=0))

    print("F1 Score:", f1_score(y_test, y_pred, zero_division=0))

    # -----------------------------------
    # CONFUSION MATRIX
    # -----------------------------------

    print("\nConfusion Matrix:\n")

    print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

    # -----------------------------------
    # CLASSIFICATION REPORT
    # -----------------------------------

    print("\nClassification Report:\n")

    print(classification_report(y_test, y_pred, labels=[0, 1], zero_division=0))

    # -----------------------------------
    # SAVE MODEL
    # -----------------------------------

    joblib.dump(model, "model.pkl")

    print("\nModel saved to model.pkl")

# -----------------------------------
# UNSUPERVISED TRAINING
# -----------------------------------

else:

    print("\nNo labels found.")

    print("Training unsupervised model...")

    # model = IsolationForest(

    #     contamination=0.2,

    #     random_state=42,

    #     n_estimators=200,
    # )

    model = IsolationForest(
        n_estimators=300,
        max_samples="auto",
        contamination=0.08,
        max_features=0.9,
        bootstrap=False,
        random_state=42,
    )

    model.fit(X)

    joblib.dump(model, "model.pkl")

    print("\nModel trained and " "saved to model.pkl")
