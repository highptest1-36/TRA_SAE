"""Diagnostic: send the exact 6 BTC quick-check samples, time each, hit localhost
to isolate model latency from the ngrok hop. Prints latency + answer + premises."""
import json, time, urllib.request, sys

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000/predict"

PREM = [
    "If a researcher completed ethics training and has lab access, then that researcher can handle participant data.",
    "If a researcher can handle participant data and has supervisor approval, then that researcher may join Study Alpha.",
    "Every researcher who may join Study Alpha is listed as an active contributor.",
    "Asha completed ethics training.",
    "Asha has lab access.",
    "Asha has supervisor approval.",
    "Study Alpha has 12 enrolled participants.",
    "No premise states whether Asha has budget approval.",
]
SAMPLES = [
    {"query_id": "quick_type1_mc", "type": "type1", "premises": PREM,
     "options": ["A", "B", "C", "D"],
     "query": "Based on the premises, which option is logically supported?\nA. Asha may join Study Alpha\nB. Asha cannot handle participant data\nC. Asha has budget approval\nD. Study Alpha has 20 enrolled participants",
     "_exp_ans": "A", "_exp_prem": [0, 1, 3, 4, 5]},
    {"query_id": "quick_type1_yes_no", "type": "type1", "premises": PREM,
     "options": ["Yes", "No", "Uncertain"],
     "query": "Is Asha listed as an active contributor?",
     "_exp_ans": "Yes", "_exp_prem": [0, 1, 2, 3, 4, 5]},
    {"query_id": "quick_type1_uncertain", "type": "type1", "premises": PREM,
     "options": ["Yes", "No", "Uncertain"],
     "query": "Does Asha have budget approval?",
     "_exp_ans": "Uncertain", "_exp_prem": [7]},
    {"query_id": "quick_type1_number", "type": "type1", "premises": PREM, "options": [],
     "query": "How many enrolled participants does Study Alpha have?",
     "_exp_ans": "12", "_exp_prem": [6]},
    {"query_id": "quick_type1_text", "type": "type1", "premises": PREM, "options": [],
     "query": "Which researcher may join Study Alpha?",
     "_exp_ans": "Asha", "_exp_prem": [0, 1, 3, 4, 5]},
    {"query_id": "quick_type2_physics", "type": "type2", "premises": [], "options": [],
     "query": "A capacitor has capacitance C = 47 μF and is connected to a potential difference U = 12 V. Calculate the energy stored in the capacitor.",
     "_exp_ans": "3.38e-3", "_exp_unit": "J"},
]


def post(obj, timeout=90):
    data = json.dumps([{k: v for k, v in obj.items() if not k.startswith("_")}]).encode()
    req = urllib.request.Request(URL, data=data, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode()), time.time() - t0


def main():
    print(f"POST → {URL}\n" + "=" * 80)
    lats = []
    for s in SAMPLES:
        try:
            body, lat = post(s)
            r = body[0] if isinstance(body, list) and body else {}
        except Exception as e:
            print(f"{s['query_id']:24} TIMEOUT/ERROR: {e}")
            continue
        lats.append(lat)
        ans = r.get("answer"); prem = r.get("premises_used"); unit = r.get("unit")
        exp_a = s.get("_exp_ans"); exp_p = s.get("_exp_prem")
        a_ok = str(ans).strip().lower() == str(exp_a).strip().lower()
        p_ok = (prem == exp_p) if exp_p is not None else "-"
        flag = ">60s!!" if lat > 60 else ("WARN" if lat > 45 else "ok")
        print(f"{s['query_id']:24} {lat:6.1f}s [{flag:6}] ans={str(ans)[:14]!r} "
              f"(exp {exp_a!r} {'OK' if a_ok else 'XX'})  unit={unit!r}  "
              f"prem={prem} (exp {exp_p} {'OK' if p_ok is True else ('XX' if p_ok is False else '-')})")
    if lats:
        print("=" * 80)
        print(f"latency: n={len(lats)} min={min(lats):.1f}s max={max(lats):.1f}s "
              f"mean={sum(lats)/len(lats):.1f}s  | >60s would FAIL BTC")


if __name__ == "__main__":
    main()
