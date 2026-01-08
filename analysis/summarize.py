
import json, sys
from pathlib import Path

p = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/paper_results.json")
data = json.loads(p.read_text())

print("Pre-registered params:", data.get("pre_registered", {}))
print("Global critical:", data.get("global_critical"))

conds = data.get("conditions", {})
for name, payload in conds.items():
    c = payload.get("collapsed", {})
    r = payload.get("rigidity", {})
    print(f"\n== {name} ==")
    print("  collapse  buf_mean:", c.get("buf_mean"), "unbuf_mean:", c.get("unbuf_mean"), "p:", c.get("p_value"))
    print("  rigidity  buf_mean:", r.get("buf_mean"), "unbuf_mean:", r.get("unbuf_mean"), "p:", r.get("p_value"))
