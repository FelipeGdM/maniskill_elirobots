import pandas as pd

DATA = "/workspaces/maniskill_elirobots/output.csv"

df = pd.read_csv(DATA)

df["target_qpos/0"] = df["agent/controller/arm/target_qpos/0"]
# print(df.describe())

headers = ["agent/qpos/0", "action/0", "target_qpos/0"]

df = df[headers]

df["test"] = df["action/0"].clip(-1, 1) * 0.5 + df["target_qpos/0"]

print(df)
