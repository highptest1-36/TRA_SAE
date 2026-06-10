r"""
Fill BTC's official notation template -> API_TRA_SAE/notation_mapping.csv.

Reads API_TRA_SAE/EXACT2026_Notation_Mapping_Template.csv (BTC's 74 canonical rows,
your_notation empty) and fills `your_notation` from the FILL map below. Keeps
BTC's exact canonical_latex/meaning rows so the regex-replace matches the dataset.

Policy (per BTC guidance — our model is an LLM that reads LaTeX fine):
  - fill operators with ASCII (\times->*, \div->/, \leq-><=, \frac{a}{b}->a/b ...)
  - fill unit/prefix Greek (\mu->u, \Omega->ohm, \degree->degree)
  - fill Greek variables with their plain names (\theta->theta ...)
  - leave already-ASCII rows blank (V, A, ohm, J, Hz, m, s, prefixes ...) → BTC
    keeps them unchanged, which is exactly what the model wants.
Output units are independently emitted in ASCII by predict_core._to_ascii_unit.
"""
from __future__ import annotations
import csv, os

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(HERE, "EXACT2026_Notation_Mapping_Template.csv")
OUT = os.path.join(HERE, "notation_mapping.csv")

FILL = {
    # operators
    r"\times": "*", r"\cdot": "*", r"\div": "/", r"\frac{a}{b}": "a/b",
    r"\pm": "+-", r"\mp": "-+", r"\approx": "~", r"\neq": "!=",
    r"\leq": "<=", r"\geq": ">=", r"\propto": "proportional to",
    r"\infty": "infinity", r"\sqrt{}": "sqrt()", r"\sqrt[n]{}": "nth-root",
    r"\times 10^{n}": "*10^", r"\%": "%", r"\angle": "angle", r"\degree": "degree",
    # calculus / vectors (rare in this dataset)
    r"\int": "integral", r"\sum": "sum", r"\partial": "d", r"\nabla": "grad",
    r"\vec{}": "vector", r"\hat{}": "unit-vector",
    # Greek — variables → names; units/prefixes → ASCII
    r"\alpha": "alpha", r"\beta": "beta", r"\gamma": "gamma",
    r"\delta": "delta", r"\Delta": "Delta", r"\epsilon": "epsilon",
    r"\varepsilon": "epsilon", r"\theta": "theta", r"\lambda": "lambda",
    r"\mu": "u", r"\pi": "pi", r"\rho": "rho", r"\sigma": "sigma",
    r"\tau": "tau", r"\phi": "phi", r"\varphi": "phi", r"\Phi": "Phi",
    r"\omega": "omega", r"\Omega": "ohm",
}


def main() -> None:
    with open(TEMPLATE, encoding="utf-8") as f:
        rows = list(csv.reader(f))
    header, data = rows[0], rows[1:]
    filled = 0
    out_rows = []
    for r in data:
        if not r:
            continue
        canon = r[0]
        meaning = r[1] if len(r) > 1 else ""
        your = FILL.get(canon, "")
        if your:
            filled += 1
        out_rows.append([canon, meaning, your])
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["canonical_latex", "meaning", "your_notation"])
        w.writerows(out_rows)
    print(f"Wrote {len(out_rows)} rows ({filled} filled, "
          f"{len(out_rows)-filled} left blank) → {OUT}")


if __name__ == "__main__":
    main()
