# Final broad cu126 versus legacy CUDA tail measurements

Serial idle-host runs, NVIDIA L4, affinity core 2, OMP threads 1, 20 warmups and 500 samples per workload/phase/run. First LBBL; reverse BLLB performed once after 30 seconds idle because initial broad p95 drift reached 8.01%. All runs retained. No further reruns or selection.

Reference: Python 3.9 legacy environment. Candidate: Python 3.11 / Torch 2.13.0+cu126 verified broad overlay. Canonical input SHA256 e57529bb1b2e4fe5dc56b6bba64c8fd9a7a8ecafebcd9b5723025e0d18ad0848. Native extension hashes before/after identical.

Ratios >1 mean broad slower. Combined ratios use mean of all four per-run quantiles per variant, not pooled sample quantiles.

| Workload | Phase | Initial median/p95 ratio | Reverse median/p95 ratio | All-run median/p95 ratio | Initial p95 drift legacy/broad | Reverse p95 drift legacy/broad |
|---|---|---:|---:|---:|---:|---:|
| traj_128x25x2_f32 | forward | 1.0231/1.0455 | 1.0180/1.0345 | 1.0205/1.0400 | -1.97%/+4.96% | -0.23%/-1.00% |
| traj_128x25x2_f32 | forward_backward | 1.0713/1.0812 | 1.0644/1.0700 | 1.0678/1.0756 | -2.78%/+1.46% | -0.65%/-1.74% |
| representative_100x10x2_f32 | forward | 1.0512/1.0520 | 1.0346/1.0173 | 1.0429/1.0345 | -3.61%/+8.01% | +0.72%/-0.77% |
| representative_100x10x2_f32 | forward_backward | 1.0843/1.0856 | 1.0768/1.0727 | 1.0805/1.0791 | -3.30%/+1.36% | -0.57%/-0.96% |
| channels_20x25x3_f32 | forward | 1.0536/1.0707 | 1.0326/1.0303 | 1.0430/1.0504 | -1.42%/+6.17% | -0.84%/-0.30% |
| channels_20x25x3_f32 | forward_backward | 1.1046/1.1161 | 1.0841/1.0777 | 1.0943/1.0967 | -2.08%/+1.28% | -1.04%/-1.07% |
| high_channel_stress_8x10x7_f32 | forward | 1.0499/1.0762 | 1.0435/1.0414 | 1.0467/1.0587 | -1.38%/+1.99% | +0.15%/+0.99% |
| high_channel_stress_8x10x7_f32 | forward_backward | 1.0916/1.1063 | 1.0794/1.0837 | 1.0854/1.0949 | -1.54%/+1.57% | +0.04%/-1.81% |

## Reproduction

Run only on an otherwise idle released GPU host. Existing interpreters and pinned overlays required.
```sh
/tmp/sigsvgd-cu126-env/bin/python /tmp/sigsvgd-broad-cu126/final_tail_measurements.py
# Only if initial same-variant p95 drift exceeds 5%:
/tmp/sigsvgd-cu126-env/bin/python /tmp/sigsvgd-broad-cu126/final_tail_measurements.py --reverse
```

All native hashes, host inventory, complete sample arrays and individual drift statistics are retained alongside this summary. Script refuses overwrite; copy it with a new output directory to reproduce rather than deleting observations.
