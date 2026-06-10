#!/usr/bin/env python3
"""
verify_paper.py — Kiểm tra tính toàn vẹn của bản Expert Systems (TRA_SAE_Wiley.tex).
Chạy bất cứ lúc nào để xác nhận file chưa hỏng sau khi sửa / sau khi Colab reconnect.

    python3 paper/bin/verify_paper.py

Không cần LaTeX. Chỉ kiểm tra cấu trúc nguồn (không thay thế việc compile trên Overleaf).
"""
import re, sys, os

BASE = "/content/drive/MyDrive/TRA-SAE/paper/format/WileyDesign/Optimal-Design-layout"
TEX  = f"{BASE}/TRA_SAE_Wiley.tex"
BIB  = f"{BASE}/references.bib"

ok = True
def check(cond, msg):
    global ok
    print(("  OK  " if cond else " FAIL ") + msg)
    ok = ok and cond

if not os.path.exists(TEX):
    print("KHONG TIM THAY:", TEX); sys.exit(1)

t = open(TEX).read()
bib = open(BIB).read()

# 1. Citation engine
check("\\documentclass[APA," in t or "\\documentclass[CHICAGO," in t,
      "documentclass dung [APA] hoac [CHICAGO]")
check("\\bibliography{references}" in t, "co \\bibliography{references} (BibTeX)")
active_bs=[l for l in t.splitlines() if l.strip().startswith("\\bibliographystyle")]
check(not active_bs, "KHONG co \\bibliographystyle active trong body (class tu lo; tranh loi 'Illegal another bibstyle')")
check("thebibliography" not in t, "KHONG con \\begin{thebibliography} thu cong")
check("\\let\\citep\\cite" in t, "co alias \\citep/\\citet (preamble)")

# 2. Abstract <= 250 tu
m = re.search(r"\\abstract\[ABSTRACT\]\{(.+?)\}\s*\n\s*\\maketitle", t, re.S)
if m:
    b = re.sub(r"\\[a-zA-Z]+\*?", " ", m.group(1))
    b = re.sub(r"[\$\{\}\\~%]", " ", b)
    nw = len([w for w in re.split(r"\s+", b) if w.strip()])
    check(nw <= 250, f"abstract = {nw} tu (gioi han 250)")
else:
    check(False, "khong parse duoc abstract")

# 3. Running title < 40
rt = re.search(r"\\titlemark\{([^}]*)\}", t)
if rt:
    check(len(rt.group(1)) < 40, f"running title = {len(rt.group(1))} ky tu (<40): {rt.group(1)!r}")

# 4. Khong con placeholder ngay thang
check("00 Month 2026" not in t, "KHONG con '00 Month 2026'")

# 5. Cite keys deu co trong .bib
keys = set()
for mm in re.finditer(r"\\cite[pt]?\{([^}]*)\}", t):
    for k in mm.group(1).split(","):
        keys.add(k.strip())
bibkeys = set(re.findall(r"@\w+\{([^,]+),", bib))
missing = sorted(k for k in keys if k not in bibkeys)
check(not missing, f"tat ca {len(keys)} cite-key co trong .bib (thieu: {missing or 'khong'})")
check("EXACT2026" in bibkeys and "Nguyen2025XAIChallenge" in bibkeys,
      "2 entry challenge moi co trong .bib")

# 6. Can bang dau ngoac / environment
check(t.count("{") == t.count("}"), f"can bang ngoac {{}} (lech {t.count('{')-t.count('}')})")
for env in ["table", "table*", "figure", "tabular*", "itemize", "lstlisting", "equation", "document"]:
    o = len(re.findall(r"\\begin\{" + re.escape(env) + r"\}", t))
    c = len(re.findall(r"\\end\{" + re.escape(env) + r"\}", t))
    if o or c:
        check(o == c, f"environment '{env}': {o} begin / {c} end")

# 7. Author info placeholder (nhac nho)
if "Anonymous Author(s)" in t:
    print("  NOTE  van con 'Anonymous Author(s)' -> ban can dien thong tin tac gia")
if "ANONYMISED-FOR-REVIEW" in t:
    print("  NOTE  van con repo placeholder 'ANONYMISED-FOR-REVIEW' -> dien link repo")

print("\n=> " + ("TAT CA OK." if ok else "CO LOI - xem dong FAIL o tren."))
sys.exit(0 if ok else 1)
