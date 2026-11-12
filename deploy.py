"""Send sample prediction to served model (mlflow models serve)."""
import sys

import requests

port = sys.argv[1] if len(sys.argv) > 1 else "1234"
columns = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"]
row = [8.3252, 41.0, 6.98, 1.02, 322.0, 2.56, 37.88, -122.23]

r = requests.post(
    f"http://localhost:{port}/invocations",
    json={"dataframe_split": {"columns": columns, "data": [row]}},
    timeout=30,
)
r.raise_for_status()
print(r.json())
