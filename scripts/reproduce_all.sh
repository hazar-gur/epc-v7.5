
#!/usr/bin/env bash
set -euo pipefail

python experiments/run_paper_suite.py --all --out results/paper_results.json
python analysis/summarize.py results/paper_results.json
