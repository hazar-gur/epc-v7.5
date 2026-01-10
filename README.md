# EPC v7.5 — Identity Binding and Amplification in Adaptive Agents

This repository contains the full code and reproduction suite for the preprint:

**“Identity Binding and Amplification as a Mechanism of Collapse in Adaptive Agents”**  
Hazar Gür (2026)

The goal of this work is to provide a **minimal, falsifiable computational mechanism** explaining when and why collapse emerges in adaptive agents under stress, and when it does not.

The repository is intended as a **transparent research artifact** for inspection, critique, and reproduction.

---

## Overview

The EPC (Externalized Policy Coherence) controller augments a simple adaptive agent with:

- a **slow identity variable** (rigidity),
- an **asymmetric identity buffer** that prevents transient stress from binding to identity,
- and a downstream **amplification pathway** that modulates threat sensitivity.

The central claim tested here is:

> **Collapse does not arise from instability alone, nor from identity rigidity itself, but from an identity-driven amplification pathway that escalates threat under sustained pressure.**

All claims are restricted to the model class studied.

---

## Repository Structure

```text
epc-v7.5/
├── src/
│   ├── epc_v75.py          # EPC v7.5 controller implementation
│   └── __init__.py
│
├── experiments/
│   └── run_paper_suite.py # Two-pass validation + negative controls
│
├── configs/
│   ├── pre_registered.yaml # Locked parameters (used in paper)
│   └── conditions.yaml     # Experimental condition definitions
│
├── analysis/
│   └── summarize.py        # Optional helper for inspecting JSON output
│
├── scripts/
│   └── reproduce_all.sh    # Convenience wrapper
│
├── results/                # Output directory (not tracked)
│
├── requirements.txt
└── README.md
```
---

## Summary of Core Results

The table below summarizes the primary behavioral outcomes reproduced by this
repository. Full statistical results (effect sizes, confidence intervals, and
p-values) are provided in the paper and in `results/paper_results.json`.

| Condition              | Buffered Collapse | Unbuffered Collapse | Interpretation |
|------------------------|-------------------|---------------------|----------------|
| burst_only             | 0%                | 100%                | Buffer absorbs transient stress |
| burst_mismatch         | 100%              | 100%                | Social pressure erodes buffering |
| mid_punishment_only    | 0%                | 100%                | Negative control (steady-state prediction confirmed) |
| support_only           | 0%                | 0%                  | Negative control (support bypasses buffer) |

These results support the central claim that collapse is not caused by instability
alone, nor by identity rigidity per se, but by an identity-driven amplification
pathway under sustained pressure.

## Reproducing the Paper Results

### 1. Set up a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

Run the full paper suite

python3 -m experiments.run_paper_suite

This executes:
	•	steady-state buffer analysis,
	•	two-pass cross-validated evaluation,
	•	all pre-registered conditions,
	•	negative controls,
	•	causal dissociation tests.

Results are written to:

results/paper_results.json

⸻

Optional: Run a Single Condition

python3 -m experiments.run_paper_suite --condition burst_only

Available conditions include:
	•	burst_only
	•	burst_mismatch
	•	mid_punishment_only (negative control)
	•	support_only (negative control)

Output

The JSON output includes:
	•	pre-registered parameters,
	•	the locked global collapse threshold,
	•	effect sizes, confidence intervals, and p-values for each condition.

Raw episode trajectories are not saved to keep the artifact minimal and interpretable.

---

Modeling Commitments

This code enforces several non-negotiable constraints, explicitly discussed in the paper:
	•	Two-pass protocol to prevent threshold endogeneity
	•	Single global threshold applied across all conditions
	•	Held-out seeds for evaluation
	•	Negative controls included by design
	•	No post-hoc parameter tuning

These choices are deliberate and documented.

---

What This Model Is — and Is Not

This model is:
	•	a mechanistic demonstration,
	•	a falsifiable computational claim,
	•	a minimal abstraction.

This model is not:
	•	a detailed brain model,
	•	a claim of biological optimality,
	•	a universal theory of collapse.

```
License

MIT License.
You are free to inspect, modify, and build upon this work, with attribution.

⸻

Contact

For questions, critique, or discussion:

Hazar Gür
(see preprint for contact details)

