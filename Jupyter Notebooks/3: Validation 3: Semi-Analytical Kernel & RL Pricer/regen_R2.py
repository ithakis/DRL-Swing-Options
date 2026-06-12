"""Regenerate ONLY episode_efficiency.csv (R2) from saved agents, now that the
kernel-on 32 768-episode agents exist. Leaves R1/R5 CSVs untouched."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gen_rl_validation as G

df = G.gen_R2()
df.to_csv(HERE / "episode_efficiency.csv", index=False)
print("\nepisode_efficiency.csv episode counts (kernel_on):")
print(df[df.method == "kernel_on"].groupby("episodes").seed.count())
print("lsm_M5_ci95:", df.lsm_M5_ci95.iloc[0])
