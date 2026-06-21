#!/usr/bin/env python3
"""Join synthetic_dataset.csv with derivatives_output.csv.

Pruning:
  - Hard crashes: rows with wrong column count, bad ID format, or non-numeric values
  - Soft crashes: rows where Newton_iterSteps >= 1500 or GMGF_iterSteps >= 1500

Output:
  - datasets/run_20260604_154509/dataset_joined.csv   (cleaned + joined)
  - datasets/run_20260604_154509/dataset_report.yaml  (stats + histogram)
"""

from __future__ import annotations

import csv
import re
import sys
from collections import Counter
from pathlib import Path

import yaml

SYNTHETIC_CSV = Path("datasets/run_20260604_154509/synthetic_dataset.csv")
DERIVATIVES_CSV = Path("datasets/derivatives_output.csv")
OUTPUT_CSV = Path("datasets/run_20260604_154509/dataset_joined.csv")
REPORT_YAML = Path("datasets/run_20260604_154509/dataset_report.yaml")

KAPPA_MIN, KAPPA_MAX = -25, 25
MAX_ITER = 1500
EXPECTED_COLS = 6
ID_PATTERN = re.compile(r"^P_[0-9a-f]+$")


# ── Derivative parser ─────────────────────────────────────────────────────────

def _extract_field(key: str, text: str) -> str | None:
    m = re.search(rf"{re.escape(key)}\s*:\s*([^,}}]+)", text)
    return m.group(1).strip() if m else None


def parse_derivatives(path: Path) -> dict[str, dict]:
    records: dict[str, dict] = {}
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip().strip('"').strip("'").strip("{}").strip()
            if not line:
                continue
            pid = _extract_field("id", line)
            if not pid:
                continue
            # Keep first occurrence per ID
            if pid in records:
                continue
            try:
                records[pid] = {
                    "x0": float(_extract_field("x0", line) or "nan"),
                    "f(x0)": float(_extract_field("f(x0)", line) or "nan"),
                    "f'(x0)": float(_extract_field("f'(x0)", line) or "nan"),
                    "f''(x0)": float(_extract_field("f''(x0)", line) or "nan"),
                    # kappa_raw from file is unrounded f''/(f')^2; we re-round + clip below
                    "kappa_raw": float(_extract_field("kappa", line) or "nan"),
                }
            except (ValueError, TypeError):
                continue
    return records


# ── Synthetic CSV reader ──────────────────────────────────────────────────────

def read_synthetic(
    path: Path,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Return (clean_rows, broken_rows, soft_crash_rows)."""
    clean: list[dict] = []
    broken: list[dict] = []
    soft: list[dict] = []

    with open(path, encoding="utf-8") as fh:
        fh.readline()  # skip header
        for line_no, raw in enumerate(fh, start=2):
            raw = raw.rstrip("\n")
            if not raw.strip():
                continue
            parts = raw.split(",")

            if len(parts) != EXPECTED_COLS:
                broken.append(
                    {"line": line_no, "raw": raw[:140], "reason": f"wrong column count ({len(parts)})"}
                )
                continue

            pid = parts[0].strip()
            if not ID_PATTERN.match(pid):
                broken.append(
                    {"line": line_no, "raw": raw[:140], "reason": f"invalid Problem_ID: {pid!r}"}
                )
                continue

            try:
                row: dict = {
                    "Problem_ID": pid,
                    "y_Target": float(parts[1]),
                    "Newton_absTime": float(parts[2]),
                    "Newton_iterSteps": int(parts[3]),
                    "GMGF_absTime": float(parts[4]),
                    "GMGF_iterSteps": int(parts[5]),
                }
            except ValueError as exc:
                broken.append(
                    {"line": line_no, "raw": raw[:140], "reason": f"parse error: {exc}"}
                )
                continue

            if row["Newton_iterSteps"] >= MAX_ITER and row["GMGF_iterSteps"] >= MAX_ITER:
                soft.append(row)
            else:
                clean.append(row)

    return clean, broken, soft


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    derivatives = parse_derivatives(DERIVATIVES_CSV)
    clean_rows, broken_rows, soft_rows = read_synthetic(SYNTHETIC_CSV)

    total_raw = len(clean_rows) + len(broken_rows) + len(soft_rows)

    # Join clean rows with derivatives — keep all clean rows, fill NaN where no match.
    joined: list[dict] = []
    no_match_ids: set[str] = set()
    no_match_rows: int = 0

    for row in clean_rows:
        pid = row["Problem_ID"]
        d = derivatives.get(pid)
        if d is None:
            no_match_ids.add(pid)
            no_match_rows += 1
            joined.append(
                {
                    **row,
                    "x0": "",
                    "f(x0)": "",
                    "f'(x0)": "",
                    "f''(x0)": "",
                    "kappa": "",
                }
            )
        else:
            kappa_clipped = max(KAPPA_MIN, min(KAPPA_MAX, round(d["kappa_raw"])))
            joined.append(
                {
                    **row,
                    "x0": d["x0"],
                    "f(x0)": d["f(x0)"],
                    "f'(x0)": d["f'(x0)"],
                    "f''(x0)": d["f''(x0)"],
                    "kappa": kappa_clipped,
                }
            )

    joined_with_deriv = [r for r in joined if r["kappa"] != ""]

    # Class distribution: which solver was faster?
    gmgf_faster = sum(1 for r in joined if r["GMGF_absTime"] < r["Newton_absTime"])
    newton_faster = sum(1 for r in joined if r["Newton_absTime"] < r["GMGF_absTime"])
    tied = len(joined) - gmgf_faster - newton_faster

    # Kappa histogram (integer values -25 … 25, only rows with derivative)
    kappa_hist: Counter[int] = Counter(r["kappa"] for r in joined_with_deriv)
    kappa_hist_full = {k: kappa_hist.get(k, 0) for k in range(KAPPA_MIN, KAPPA_MAX + 1)}

    def pct(n: int, total: int) -> float:
        return round(n / total * 100, 2) if total else 0.0

    # ── YAML report ───────────────────────────────────────────────────────────
    report: dict = {
        "dataset_summary": {
            "total_raw_rows": total_raw,
            "hard_crashes_removed": len(broken_rows),
            "soft_crashes_removed": len(soft_rows),
            "no_derivative_match_removed_ids": len(no_match_ids),
        "no_derivative_match_removed_rows": no_match_rows,
            "final_rows": len(joined),
        },
        "quotes": {
            "hard_crashes": {
                "count": len(broken_rows),
                "percent_of_total": pct(len(broken_rows), total_raw),
            },
            "soft_crashes": {
                "count": len(soft_rows),
                "percent_of_total": pct(len(soft_rows), total_raw),
            },
            "no_derivative_match": {
                "unique_problem_ids": len(no_match_ids),
                "rows_removed": no_match_rows,
                "avg_rows_per_id": round(no_match_rows / len(no_match_ids), 1) if no_match_ids else 0,
                "percent_of_clean_rows": pct(no_match_rows, len(clean_rows)),
            },
            "final_dataset": {
                "count": len(joined),
                "percent_of_total": pct(len(joined), total_raw),
                "rows_with_derivatives": len(joined_with_deriv),
                "rows_without_derivatives": no_match_rows,
            },
        },
        "soft_crashes": {
            "problem_ids": sorted({r["Problem_ID"] for r in soft_rows}),
        },
        "class_distribution": {
            "description": "Based on absTime comparison (lower is faster)",
            "gmgf_faster": {
                "count": gmgf_faster,
                "percent": pct(gmgf_faster, len(joined)),
            },
            "newton_faster": {
                "count": newton_faster,
                "percent": pct(newton_faster, len(joined)),
            },
            "tied": {
                "count": tied,
                "percent": pct(tied, len(joined)),
            },
        },
        "kappa_histogram": {
            "description": "Count per integer kappa value (clipped to [-25, 25])",
            "values": kappa_hist_full,
        },
    }

    REPORT_YAML.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_YAML, "w", encoding="utf-8") as fh:
        yaml.dump(report, fh, default_flow_style=False, allow_unicode=True, sort_keys=False)

    # ── Write joined CSV ───────────────────────────────────────────────────────
    fieldnames = [
        "Problem_ID", "y_Target",
        "Newton_absTime", "Newton_iterSteps",
        "GMGF_absTime", "GMGF_iterSteps",
        "x0", "f(x0)", "f'(x0)", "f''(x0)", "kappa",
    ]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in joined:
            writer.writerow({k: row[k] for k in fieldnames})

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"Total raw rows:          {total_raw}")
    print(f"Hard crashes removed:    {len(broken_rows):>5}  ({pct(len(broken_rows), total_raw):.1f}%)")
    print(f"Soft crashes removed:    {len(soft_rows):>5}  ({pct(len(soft_rows), total_raw):.1f}%)")
    print(f"No derivative match:     {len(no_match_ids):>5}  unique IDs")
    print(f"Final rows:              {len(joined):>5}  ({pct(len(joined), total_raw):.1f}%)")
    print()
    print(f"Class distribution (n={len(joined)}):")
    print(f"  GMGF faster:  {gmgf_faster:>5}  ({pct(gmgf_faster, len(joined)):.1f}%)")
    print(f"  Newton faster:{newton_faster:>5}  ({pct(newton_faster, len(joined)):.1f}%)")
    print(f"  Tied:         {tied:>5}  ({pct(tied, len(joined)):.1f}%)")
    print()
    print(f"Output CSV:  {OUTPUT_CSV}")
    print(f"Report:      {REPORT_YAML}")


if __name__ == "__main__":
    main()
