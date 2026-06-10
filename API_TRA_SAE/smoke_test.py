"""
Smoke test for the TRA-SAE /predict API — mimics the EXACT 2026 BTC grading.

Sends N samples (type1 + type2) ONE AT A TIME using the BTC request schema, then:
  - validates the response is a JSON list with the 6 required fields,
  - asserts `explanation` is non-empty, type1 unit=="" + premises_used is a list
    of 0-based ints, type2 premises_used==[],
  - if options present, asserts answer is exactly one option,
  - measures per-query latency and FAILS any query over 60s,
  - scores against ground truth (debug only; dataset has known unit bugs).

Usage:
  python API_TRA_SAE/smoke_test.py --url https://<host>/predict --n 10 --gap 4
"""
from __future__ import annotations
import argparse, csv, json, os, sys, time, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGIC = os.path.join(ROOT, "data/Logic_Based_Educational_Queries_Text_Only/Logic_Based_Educational_Queries.json")
PHYS = os.path.join(ROOT, "data/Physics_Problems_Text_Only/Physics_Problems_Text_Only.csv")
REQUIRED = {"query_id", "answer", "unit", "explanation", "premises_used", "reasoning"}
LATENCY_LIMIT = 60.0


def _sleep(s):
    if s > 0:
        time.sleep(s)


def _post(url, obj, timeout=90):
    data = json.dumps([obj]).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8")), time.time() - t0


def _load_samples(n):
    half = max(1, n // 2)
    samples = []
    # type2 (physics)
    with open(PHYS, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("answer", "").strip():
                samples.append({"query_id": f"T2_{row['id']}", "type": "type2",
                                "query": row["question"], "premises": [], "options": [],
                                "_gt": row["answer"].strip(), "_unit": row.get("unit", "").strip()})
            if len(samples) >= half:
                break
    # type1 (logic) — give Yes/No/Uncertain options when the answer is one of those
    recs = json.load(open(LOGIC, encoding="utf-8"))
    cnt = 0
    for ri, rec in enumerate(recs):
        for qi, (q, a) in enumerate(zip(rec.get("questions", []), rec.get("answers", []))):
            a = str(a)
            opts = ["Yes", "No", "Uncertain"] if a.lower() in ("yes", "no", "unknown", "uncertain") else []
            samples.append({"query_id": f"T1_{ri}_{qi}", "type": "type1", "query": q,
                            "premises": rec.get("premises-NL", []), "options": opts,
                            "_gt": ("Uncertain" if a.lower() == "unknown" else a), "_unit": ""})
            cnt += 1
            break
        if cnt >= (n - half):
            break
    return samples[:n]


def _score(pred, s):
    gt, pa = s["_gt"].strip().lower(), str(pred.get("answer", "")).strip().lower()
    if s["type"] == "type1":
        return pa == gt or gt in pa or (pa and pa in gt)
    try:
        pv, gv = float(pa.split()[0]), float(gt.split()[0])
        ok = abs(pv - gv) <= max(0.01, 0.02 * abs(gv))
        return ok
    except Exception:
        return pa == gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000/predict")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--gap", type=float, default=4.0)
    a = ap.parse_args()
    samples = _load_samples(a.n)
    print(f"Testing {len(samples)} samples → {a.url}  (gap={a.gap}s)\n")
    schema_ok = lat_ok = correct = 0
    max_lat = 0.0
    for i, s in enumerate(samples, 1):
        payload = {k: v for k, v in s.items() if not k.startswith("_")}
        try:
            body, lat = _post(a.url, payload)
        except Exception as e:
            print(f"[{i}] {s['query_id']:>12} ERROR: {e}")
            _sleep(a.gap); continue
        max_lat = max(max_lat, lat)
        r = body[0] if isinstance(body, list) and body else {}
        miss = REQUIRED - set(r)
        ssok = (not miss) and bool(str(r.get("explanation", "")).strip())
        # BTC field rules
        if s["type"] == "type1":
            ssok = ssok and r.get("unit", "") == "" and isinstance(r.get("premises_used"), list)
            if s["options"]:
                ssok = ssok and r.get("answer") in s["options"]
        else:
            ssok = ssok and r.get("premises_used") == []
        lok = lat <= LATENCY_LIMIT
        cor = _score(r, s)
        schema_ok += ssok; lat_ok += lok; correct += cor
        fl = ("S" if ssok else "s") + ("T" if lok else "t") + ("C" if cor else "c")
        print(f"[{i}] {s['query_id']:>12} {s['type']} {lat:5.1f}s [{fl}] "
              f"ans={str(r.get('answer'))[:18]!r} unit={r.get('unit')!r} "
              f"prem={r.get('premises_used')}" + (f" MISS={miss}" if miss else ""))
        _sleep(a.gap)
    n = len(samples)
    print(f"\n── schema {schema_ok}/{n} | latency<=60s {lat_ok}/{n} | correct {correct}/{n} | max {max_lat:.1f}s")
    if schema_ok < n or lat_ok < n:
        print("FAIL: schema/latency issue."); sys.exit(1)
    print("PASS: BTC schema + latency OK.")


if __name__ == "__main__":
    main()
