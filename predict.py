# import requests
# import pandas as pd
# import joblib

# # Load trained model
# model = joblib.load("model.pkl")

# url = "http://localhost:9090/api/v1/query"
# query = "rate(container_cpu_usage_seconds_total[1m])"

# response = requests.get(url, params={"query": query})
# data = response.json()

# results = data['data']['result']

# for item in results:
#     value = float(item['value'][1])
    
#     prediction = model.predict([[value]])
    
#     if prediction[0] == 1:
#         print("⚠️ Potential Failure Detected!")
#     else:
#         print("✅ System Normal")

import joblib
import pandas as pd
from metrics import collect_metrics

model = joblib.load("model.pkl")


data = collect_metrics()

df = pd.DataFrame([data])

if df.empty:
    raise ValueError("No metrics collected for prediction")

# prediction = model.predict(df)

# if prediction[0] == -1:
#     print("🚨 Anomaly Detected! Possible Failure")
# else:
#     print("✅ System Normal")

print(df)