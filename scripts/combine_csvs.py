import pandas as pd


df2 = pd.read_csv("eye_tracking_results_IDA.csv")
df3 = pd.read_csv("eye_tracking_results_IAN.csv")
df4 = pd.read_csv("eye_tracking_results_MARIA.csv")
df5 = pd.read_csv("eye_tracking_results_DORIAN.csv")

combined = pd.concat([df2, df3, df4, df5], ignore_index=True)

combined.to_csv("combined_correction_data2.csv", index=False)

print("Combined dataset saved!")