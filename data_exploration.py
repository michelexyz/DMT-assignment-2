import pandas as pd
import time

train_path = "data/training_set_VU_DM.csv"

# for testing:
# train_path = "data/small_dataset.csv"

start_time = time.time()
chunk = pd.read_csv(train_path, chunksize=1000000)
print(f"load chunk: {time.time() - start_time:.2f}s")

start_time = time.time()
df = pd.concat(chunk)
print(f"concat df: {time.time() - start_time:.2f}s")

# print(df.head())
# print(df.columns)

stats = []

for column in df.columns:
    stats.append(df[column].describe())

df_stats = pd.concat(stats, axis=1).T

stats_path = f"stats_and_graphs/stats/{train_path.split('/')[1][:-4]}_stats.csv"
df_stats.to_csv(stats_path)
