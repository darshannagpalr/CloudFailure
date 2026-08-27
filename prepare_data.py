# import os
# import time
# import argparse
# from statistics import median

# import pandas as pd

# from metrics import collect_metrics

# FEATURES = ("cpu", "memory", "restarts")
# BASELINE_WINDOW = int(os.environ.get("BASELINE_WINDOW", "10"))
# CPU_MULTIPLIER = float(os.environ.get("CPU_MULTIPLIER", "1.25"))
# MEMORY_MULTIPLIER = float(os.environ.get("MEMORY_MULTIPLIER", "1.25"))
# RESTART_DELTA = float(os.environ.get("RESTART_DELTA", "1"))


# def build_baseline(rows):
#     return {
#         "cpu": median(row["cpu"] for row in rows),
#         "memory": median(row["memory"] for row in rows),
#         "restarts": median(row["restarts"] for row in rows),
#     }


# def label(row, baseline):
#     if (
#         row["cpu"] > baseline["cpu"] * CPU_MULTIPLIER
#         or row["memory"] > baseline["memory"] * MEMORY_MULTIPLIER
#         or row["restarts"] >= baseline["restarts"] + RESTART_DELTA
#     ):
#         return 1
#     return 0


# def main():
#     parser = argparse.ArgumentParser(description="Collect Prometheus metrics and label them for training.")
#     parser.add_argument("--samples", type=int, default=60, help="Number of rows to collect")
#     parser.add_argument("--interval", type=float, default=5.0, help="Seconds to wait between samples")
#     parser.add_argument("--output", default="data.csv", help="Output CSV path")
#     parser.add_argument("--reset", action="store_true", help="Overwrite the output file instead of appending")
#     args = parser.parse_args()

#     print("Collecting data...")

#     healthy_rows = []
#     label_counts = {0: 0, 1: 0}

#     if args.reset and os.path.exists(args.output):
#         os.remove(args.output)

#     for _ in range(args.samples):
#         data = collect_metrics()

#         if not all(feature in data for feature in FEATURES):
#             print("Skipping incomplete row")
#             time.sleep(5)
#             continue

#         if len(healthy_rows) < BASELINE_WINDOW:
#             failure = 0
#             healthy_rows.append(data)
#         else:
#             baseline = build_baseline(healthy_rows)
#             failure = label(data, baseline)
#             if failure == 0:
#                 healthy_rows.append(data)
#                 healthy_rows = healthy_rows[-BASELINE_WINDOW:]

#         df = pd.DataFrame([data])
#         df["failure"] = failure
#         label_counts[failure] += 1

#         print(df)

#         df.to_csv(
#             args.output,
#             mode="a",
#             header=not os.path.exists(args.output),
#             index=False,
#         )

#         time.sleep(args.interval)

#     print(f"Label distribution collected: {label_counts}")


# if __name__ == "__main__":
#     main()



import os
import time
import argparse
from statistics import median

import pandas as pd

from metrics import collect_metrics

# -----------------------------------
# FEATURES USED FOR ML
# -----------------------------------

FEATURES = (
    "cpu_usage",
    "memory_working_set",
    "restart_count",
    "oom_events",
    "cpu_pressure",
    "io_pressure",
)

# -----------------------------------
# BASELINE SETTINGS
# -----------------------------------

BASELINE_WINDOW = int(
    os.environ.get("BASELINE_WINDOW", "10")
)

CPU_MULTIPLIER = float(
    os.environ.get("CPU_MULTIPLIER", "1.25")
)

MEMORY_MULTIPLIER = float(
    os.environ.get("MEMORY_MULTIPLIER", "1.25")
)

RESTART_DELTA = float(
    os.environ.get("RESTART_DELTA", "1")
)

OOM_DELTA = float(
    os.environ.get("OOM_DELTA", "1")
)

PRESSURE_MULTIPLIER = float(
    os.environ.get("PRESSURE_MULTIPLIER", "1.5")
)

# -----------------------------------
# BASELINES PER POD
# -----------------------------------

pod_healthy_rows = {}

# -----------------------------------
# BUILD BASELINE
# -----------------------------------

def build_baseline(rows):

    return {

        "cpu_usage":
            median(row["cpu_usage"] for row in rows),

        "memory_working_set":
            median(row["memory_working_set"] for row in rows),

        "restart_count":
            median(row["restart_count"] for row in rows),

        "oom_events":
            median(row["oom_events"] for row in rows),

        "cpu_pressure":
            median(row["cpu_pressure"] for row in rows),

        "io_pressure":
            median(row["io_pressure"] for row in rows),
    }

# -----------------------------------
# LABEL FAILURE
# -----------------------------------

def label(row, baseline):

    # Hard Kubernetes failure
    if row.get("pod_failed", 0) > 0:
        return 1

    # Restart anomaly
    if (
        row["restart_count"]
        >= baseline["restart_count"] + RESTART_DELTA
    ):
        return 1

    # OOM anomaly
    if (
        row["oom_events"]
        >= baseline["oom_events"] + OOM_DELTA
    ):
        return 1

    # CPU anomaly
    if (
        row["cpu_usage"]
        > baseline["cpu_usage"] * CPU_MULTIPLIER
    ):
        return 1

    # Memory anomaly
    if (
        row["memory_working_set"]
        > baseline["memory_working_set"]
        * MEMORY_MULTIPLIER
    ):
        return 1

    # CPU pressure anomaly
    if (
        row["cpu_pressure"]
        > baseline["cpu_pressure"]
        * PRESSURE_MULTIPLIER
    ):
        return 1

    # IO pressure anomaly
    if (
        row["io_pressure"]
        > baseline["io_pressure"]
        * PRESSURE_MULTIPLIER
    ):
        return 1

    return 0

# -----------------------------------
# MAIN
# -----------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Collect Prometheus metrics "
            "for all pods and label them "
            "for ML training."
        )
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=60,
        help="Number of collection cycles"
    )

    parser.add_argument(
        "--interval",
        type=float,
        default=5.0,
        help="Seconds between samples"
    )

    parser.add_argument(
        "--output",
        default="data.csv",
        help="Output CSV file"
    )

    parser.add_argument(
        "--reset",
        action="store_true",
        help="Overwrite existing CSV"
    )

    args = parser.parse_args()

    print("Collecting pod metrics...")

    label_counts = {
        0: 0,
        1: 0
    }

    # Reset CSV
    if args.reset and os.path.exists(args.output):
        os.remove(args.output)

    # -----------------------------------
    # COLLECTION LOOP
    # -----------------------------------

    for _ in range(args.samples):

        rows = collect_metrics()

        for data in rows:

            pod_name = data["pod"]

            # Initialize pod baseline history
            if pod_name not in pod_healthy_rows:
                pod_healthy_rows[pod_name] = []

            healthy_rows = pod_healthy_rows[pod_name]

            # Ensure features exist
            if not all(
                feature in data
                for feature in FEATURES
            ):
                print(
                    f"Skipping incomplete row "
                    f"for {pod_name}"
                )
                continue

            # -----------------------------------
            # BASELINE BUILDING
            # -----------------------------------

            if len(healthy_rows) < BASELINE_WINDOW:

                failure = 0

                healthy_rows.append(data)

            else:

                baseline = build_baseline(
                    healthy_rows
                )

                failure = label(
                    data,
                    baseline
                )

                # Update healthy baseline
                if failure == 0:

                    healthy_rows.append(data)

                    healthy_rows[:] = healthy_rows[
                        -BASELINE_WINDOW:
                    ]

            # -----------------------------------
            # SAVE ROW
            # -----------------------------------

            df = pd.DataFrame([data])

            df["failure"] = failure

            label_counts[failure] += 1

            print(df)

            df.to_csv(
                args.output,
                mode="a",
                header=not os.path.exists(args.output),
                index=False,
            )

        time.sleep(args.interval)

    # -----------------------------------
    # SUMMARY
    # -----------------------------------

    print(
        f"\nLabel distribution collected: "
        f"{label_counts}"
    )

# -----------------------------------
# ENTRY POINT
# -----------------------------------

if __name__ == "__main__":
    main()