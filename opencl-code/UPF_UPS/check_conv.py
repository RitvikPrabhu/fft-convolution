#!/usr/bin/env python3
"""
Compare CPU / GPU convolution results against SciPy reference
and plot them.
"""
import numpy as np, matplotlib.pyplot as plt
from scipy.signal import fftconvolve

def load(fn): return np.fromfile(fn, dtype=np.float32)

x   = load("input.bin")
h   = load("fir.bin")
cpu = load("cpu.bin")
gpu = load("gpu.bin")

ref = fftconvolve(x, h, mode="full").astype(np.float32)

for tag, vec in (("cpu", cpu), ("gpu", gpu)):
    # trim to common length
    m = min(len(vec), len(ref))
    diff = vec[:m] - ref[:m]
    print(f"{tag:3s} :  max|err| {np.max(np.abs(diff)):.3e}   "
          f"RMS = {np.sqrt(np.mean(diff*diff)):.3e}")

# -------- quick graph ---------------------------------------------------
t = np.arange(len(ref))
plt.plot(t, ref, label="SciPy", lw=1)
plt.plot(t[:len(gpu)], gpu, "--",  label="GPU")
plt.plot(t[:len(cpu)], cpu, ":",  label="CPU")
plt.xlim(0, 8192); plt.legend(); plt.title("UPF-UPS convolution")
plt.savefig('correctness.png')
plt.show()
