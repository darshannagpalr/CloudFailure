import pandas as pd
from metrics import collect_metrics

row = collect_metrics()

df = pd.DataFrame([row])

print(df)