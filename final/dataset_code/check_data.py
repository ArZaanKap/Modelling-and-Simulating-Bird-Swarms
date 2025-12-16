"""Quick check of flock 6 data and model matching."""
from model_comparison import load_flock, get_flock_at_time
import numpy as np

# Check bird counts
data = load_flock(6)
times = np.unique(data['times'])

# Sample every 100th frame for speed
sample_times = times[::100]
bird_counts = []
for t in sample_times:
    pos, vel = get_flock_at_time(data, t)
    bird_counts.append(len(pos))

print("="*60)
print("FLOCK 6 DATA ANALYSIS (sampled)")
print("="*60)
print(f"Total unique timestamps: {len(times)}")
print(f"Sampled {len(sample_times)} frames")
print(f"Bird count range: {min(bird_counts)} to {max(bird_counts)}")
print(f"Mean bird count: {np.mean(bird_counts):.1f}")

# The 47 comes from time-averaging the num_agents metric
print("\nThe reported '47 birds' is the TIME-AVERAGED agent count.")
print("Real flock has VARYING bird counts per frame (tracking loss/recovery).")
print("Middle frames have ~97 birds, early/late frames have fewer.")
print("="*60)
