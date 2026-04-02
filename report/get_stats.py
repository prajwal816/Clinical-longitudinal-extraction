import json
from pathlib import Path

repo = Path(__file__).resolve().parents[1]
labels_dir = repo / "Data" / "train" / "labels"
train_dir = repo / "Data" / "train"
dev_dir = repo / "Data" / "dev"

# Notes per patient
for name, d in [("TRAIN", train_dir), ("DEV", dev_dir)]:
    pats = sorted([p for p in d.iterdir() if p.is_dir() and p.name.startswith("patient")])
    print(f"\n{name} SET ({len(pats)} patients):")
    for p in pats:
        n = len(list(p.glob("text_*.md")))
        print(f"  {p.name}: {n} notes")

# Label stats
total_conds = 0
cats = {}
subs = {}
statuses = {}
ev_counts = []
conds_per_pat = []
onset_has = 0
onset_null = 0

for lf in sorted(labels_dir.glob("patient_*.json")):
    data = json.loads(lf.read_text(encoding="utf-8"))
    cs = data.get("conditions", [])
    conds_per_pat.append(len(cs))
    total_conds += len(cs)
    print(f"\n{data['patient_id']}: {len(cs)} conditions")
    for c in cs:
        cats[c["category"]] = cats.get(c["category"], 0) + 1
        subs[c["subcategory"]] = subs.get(c["subcategory"], 0) + 1
        statuses[c["status"]] = statuses.get(c["status"], 0) + 1
        ev_counts.append(len(c.get("evidence", [])))
        if c.get("onset"):
            onset_has += 1
        else:
            onset_null += 1

print(f"\n{'='*60}")
print(f"TOTAL CONDITIONS: {total_conds}")
print(f"CONDITIONS PER PATIENT: {conds_per_pat}")
print(f"AVG: {sum(conds_per_pat)/len(conds_per_pat):.1f}")
print(f"\nCATEGORIES:")
for k, v in sorted(cats.items(), key=lambda x: -x[1]):
    print(f"  {k}: {v}")
print(f"\nTOP SUBCATEGORIES:")
for k, v in sorted(subs.items(), key=lambda x: -x[1])[:15]:
    print(f"  {k}: {v}")
print(f"\nSTATUSES: {statuses}")
print(f"EVIDENCE: min={min(ev_counts)} max={max(ev_counts)} avg={sum(ev_counts)/len(ev_counts):.1f} total={sum(ev_counts)}")
print(f"ONSET: has_date={onset_has} null={onset_null}")
