import pandas as pd
import time

train_path = "data/training_set_VU_DM.csv"

start_time = time.time()
chunk = pd.read_csv(train_path, chunksize=1000000)
print(f"load chunk: {time.time() - start_time:.2f}s")

start_time = time.time()
df = pd.concat(chunk)
print(f"concat df: {time.time() - start_time:.2f}s")

# print(df.head())
# print(df.columns)

for column in df.columns:
    print(df[column].describe())

