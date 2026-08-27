# import sys

# import joblib
# import pandas as pd
# from sklearn.metrics import classification_report, confusion_matrix


# def main():
#     data_path = sys.argv[1] if len(sys.argv) > 1 else "data.csv"

#     df = pd.read_csv(data_path)

#     if "cpu" not in df.columns and "cpu_usage" in df.columns:
#         df = df.rename(columns={"cpu_usage": "cpu"})

#     required_columns = ["cpu", "memory", "restarts"]
#     missing_columns = [column for column in required_columns if column not in df.columns]
#     if missing_columns:
#         raise ValueError(f"Missing required columns in {data_path}: {missing_columns}")

#     model = joblib.load("model.pkl")
#     predictions = model.predict(df[required_columns])

#     if "failure" in df.columns:
#         predicted_labels = [1 if value == -1 else 0 for value in predictions]
#         label_counts = df["failure"].value_counts().to_dict()
#         print(f"Label distribution: {label_counts}")

#         if len(label_counts) < 2:
#             print("Warning: data.csv contains only one class, so evaluation is not representative.")
#             print(f"Predicted anomalies: {sum(1 for value in predictions if value == -1)} / {len(predictions)}")
#             return

#         print("Confusion matrix:")
#         print(confusion_matrix(df["failure"], predicted_labels, labels=[0, 1]))
#         print()
#         print("Classification report:")
#         print(classification_report(df["failure"], predicted_labels, zero_division=0, labels=[0, 1]))
#     else:
#         anomaly_count = sum(1 for value in predictions if value == -1)
#         print(f"Predicted anomalies: {anomaly_count} / {len(predictions)}")


# if __name__ == "__main__":
#     main()


import sys

import joblib
import pandas as pd

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)

# -----------------------------------
# FEATURES USED FOR PREDICTION
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
# MAIN
# -----------------------------------

def main():

    data_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "data.csv"
    )

    # -----------------------------------
    # LOAD DATA
    # -----------------------------------

    df = pd.read_csv(data_path)

    print(f"\nLoaded {len(df)} rows")

    # -----------------------------------
    # VALIDATE FEATURES
    # -----------------------------------

    missing_columns = [

        column

        for column in FEATURE_COLUMNS

        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            f"\nMissing required columns "
            f"in {data_path}: "
            f"{missing_columns}"
        )

    # -----------------------------------
    # LOAD MODEL
    # -----------------------------------

    model = joblib.load("model.pkl")

    # -----------------------------------
    # PREDICT
    # -----------------------------------

    predictions = model.predict(
        df[FEATURE_COLUMNS]
    )

    # Isolation Forest:
    # -1 = anomaly
    #  1 = normal

    predicted_labels = [

        1 if value == -1 else 0

        for value in predictions
    ]

    # -----------------------------------
    # SHOW POD-LEVEL RESULTS
    # -----------------------------------

    results_df = df.copy()

    results_df["predicted_failure"] = (
        predicted_labels
    )

    # Show only important columns
    display_columns = [

        "pod",

        "cpu_usage",

        "memory_working_set",

        "restart_count",

        "oom_events",

        "cpu_pressure",

        "io_pressure",

        "predicted_failure",
    ]

    if "failure" in results_df.columns:

        display_columns.append("failure")

    print("\nPrediction Results:\n")

    print(
        results_df[
            display_columns
        ]
    )

    # -----------------------------------
    # EVALUATION
    # -----------------------------------

    if "failure" in df.columns:

        label_counts = (
            df["failure"]
            .value_counts()
            .to_dict()
        )

        print(
            f"\nLabel distribution: "
            f"{label_counts}"
        )

        # Need both classes
        if len(label_counts) < 2:

            print(
                "\nWarning: "
                "Dataset contains only "
                "one class."
            )

            anomaly_count = sum(
                predicted_labels
            )

            print(
                f"Predicted anomalies: "
                f"{anomaly_count} / "
                f"{len(predictions)}"
            )

            return

        # -----------------------------------
        # CONFUSION MATRIX
        # -----------------------------------

        print("\nConfusion Matrix:\n")

        print(

            confusion_matrix(

                df["failure"],

                predicted_labels,

                labels=[0, 1]
            )
        )

        # -----------------------------------
        # CLASSIFICATION REPORT
        # -----------------------------------

        print("\nClassification Report:\n")

        print(

            classification_report(

                df["failure"],

                predicted_labels,

                labels=[0, 1],

                zero_division=0
            )
        )

    else:

        anomaly_count = sum(
            predicted_labels
        )

        print(
            f"\nPredicted anomalies: "
            f"{anomaly_count} / "
            f"{len(predictions)}"
        )

# -----------------------------------
# ENTRY POINT
# -----------------------------------

if __name__ == "__main__":
    main()