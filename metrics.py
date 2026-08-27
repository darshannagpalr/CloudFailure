# import requests

# import os

# PROM_URL = os.environ.get("PROM_URL", "http://localhost:9090/api/v1/query")

# # Optional: narrow metrics to a specific pod or selector to avoid dilution when
# # Prometheus returns many series. Set TARGET_SELECTOR to a PromQL selector
# # fragment like: pod=~"test-app.*" or namespace="default",pod=~"test-app.*"
# TARGET_SELECTOR = os.environ.get("TARGET_SELECTOR", "")
# WINDOW = os.environ.get("PROM_WINDOW", "5m")

# # Build queries that sum over the selector when provided so the collector
# # observes the service-level usage instead of averaging many unrelated pods.
# if TARGET_SELECTOR:
#     QUERIES = {
#         "cpu": f"sum(rate(container_cpu_usage_seconds_total{{{TARGET_SELECTOR}}}[{WINDOW}]))",
#         "memory": f"sum(container_memory_working_set_bytes{{{TARGET_SELECTOR}}})",
#         "restarts": f"max(kube_pod_container_status_restarts_total{{{TARGET_SELECTOR}}})",
#     }
# else:
#     QUERIES = {
#         "cpu": f"sum(rate(container_cpu_usage_seconds_total[{WINDOW}]))",
#         "memory": "sum(container_memory_working_set_bytes)",
#         "restarts": "max(kube_pod_container_status_restarts_total)",
#     }


# def fetch_values(query):
#     try:
#         response = requests.get(PROM_URL, params={"query": query}, timeout=10)
#         response.raise_for_status()
#         payload = response.json()
#         results = payload.get("data", {}).get("result", [])
#     except (requests.RequestException, ValueError, KeyError) as error:
#         print(f"Error fetching {query}: {error}")
#         return []

#     values = []
#     for item in results:
#         try:
#             values.append(float(item["value"][1]))
#         except (KeyError, TypeError, ValueError):
#             continue
#     return values


# def collect_metrics():
#     row = {}

#     for key, query in QUERIES.items():
#         values = fetch_values(query)

#         # When queries return a single numeric result (sum/max), take the
#         # first value; otherwise fall back to average of returned series.
#         if not values:
#             row[key] = 0
#             continue

#         if len(values) == 1:
#             row[key] = values[0]
#             continue

#         if key == "restarts":
#             row[key] = max(values)
#         else:
#             row[key] = sum(values) / len(values)

#     return row



import requests
import os

PROM_URL = os.environ.get(
    "PROM_URL",
    "http://localhost:9090/api/v1/query"
)

TARGET_SELECTOR = os.environ.get(
    "TARGET_SELECTOR",
    'pod=~"test-app.*"'
)

WINDOW = os.environ.get(
    "PROM_WINDOW",
    "5m"
)

# ----------------------------------------
# QUERIES (PER POD)
# ----------------------------------------

QUERIES = {

    "cpu_usage": f'''
        sum(
            rate(
                container_cpu_usage_seconds_total{{{TARGET_SELECTOR}}}[{WINDOW}]
            )
        ) by (pod)
    ''',

    "memory_working_set": f'''
        sum(
            container_memory_working_set_bytes{{{TARGET_SELECTOR}}}
        ) by (pod)
    ''',

    "restart_count": f'''
        max(
            kube_pod_container_status_restarts_total{{{TARGET_SELECTOR}}}
        ) by (pod)
    ''',

    "oom_events": f'''
        sum(
            container_oom_events_total{{{TARGET_SELECTOR}}}
        ) by (pod)
    ''',

    "cpu_pressure": f'''
        sum(
            container_pressure_cpu_waiting_seconds_total{{{TARGET_SELECTOR}}}
        ) by (pod)
    ''',

    "io_pressure": f'''
        sum(
            container_pressure_io_waiting_seconds_total{{{TARGET_SELECTOR}}}
        ) by (pod)
    ''',

    "pod_failed": f'''
        max(
            kube_pod_status_phase{{
                {TARGET_SELECTOR},
                phase="Failed"
            }}
        ) by (pod)
    '''
}


# ----------------------------------------
# FETCH SERIES
# ----------------------------------------

def fetch_series(query):

    try:

        response = requests.get(
            PROM_URL,
            params={"query": query},
            timeout=10
        )

        response.raise_for_status()

        payload = response.json()

        return payload.get("data", {}).get("result", [])

    except Exception as error:

        print(f"Error fetching query:\n{query}\n{error}")

        return []


# ----------------------------------------
# COLLECT METRICS PER POD
# ----------------------------------------

def collect_metrics():

    pod_rows = {}

    for metric_name, query in QUERIES.items():

        results = fetch_series(query)

        for item in results:

            pod = item["metric"].get("pod")

            if not pod:
                continue

            value = float(item["value"][1])

            if pod not in pod_rows:
                pod_rows[pod] = {
                    "pod": pod
                }

            pod_rows[pod][metric_name] = value

    # Fill missing metrics with 0
    for pod in pod_rows:

        for metric in QUERIES.keys():

            if metric not in pod_rows[pod]:
                pod_rows[pod][metric] = 0

    return list(pod_rows.values())