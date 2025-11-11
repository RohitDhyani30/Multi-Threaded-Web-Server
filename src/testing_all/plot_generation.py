import pandas as pd
import matplotlib.pyplot as plt

s = pd.read_csv("C:/Users/mohit/OneDrive/Desktop/multithreaded/src/testing_all/client_log_static.csv")
d = pd.read_csv("C:/Users/mohit/OneDrive/Desktop/multithreaded/src/testing_all/client_log_dynamic.csv")

s["Timestamp"] = pd.to_datetime(s["Timestamp"], unit="ms")
d["Timestamp"] = pd.to_datetime(d["Timestamp"], unit="ms")

s = s.sort_values("Timestamp")
d = d.sort_values("Timestamp")

s["LatencySmooth"] = s["LatencyMs"].rolling(50).mean()
d["LatencySmooth"] = d["LatencyMs"].rolling(50).mean()

plt.figure(figsize=(12,4))
plt.plot(s["Timestamp"], s["LatencySmooth"])
plt.plot(d["Timestamp"], d["LatencySmooth"])
plt.xlabel("Time")
plt.ylabel("Latency (ms)")
plt.legend(["Static","Dynamic"])
plt.tight_layout()
plt.show()

s_err = s.resample("10S", on="Timestamp")["Error"].sum()
d_err = d.resample("10S", on="Timestamp")["Error"].sum()

plt.figure(figsize=(12,4))
plt.plot(s_err.index, s_err.values)
plt.plot(d_err.index, d_err.values)
plt.xlabel("Time")
plt.ylabel("Errors")
plt.legend(["Static","Dynamic"])
plt.tight_layout()
plt.show()

ta = pd.read_csv("C:/Users/mohit/OneDrive/Desktop/multithreaded/src/testing_all/thread_analysis.csv")
ta["Timestamp"] = pd.to_datetime(ta["Timestamp"], unit="ms")
ta = ta.sort_values("Timestamp")

static_value = 50
static_ts = ta["Timestamp"]

plt.figure(figsize=(12,4))
plt.plot(static_ts, [static_value]*len(static_ts))
plt.plot(ta["Timestamp"], ta["PoolSize"])
plt.xlabel("Time")
plt.ylabel("Pool Size")
plt.legend(["Static","Dynamic"])
plt.tight_layout()
plt.show()
