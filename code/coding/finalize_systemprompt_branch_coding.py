#!/usr/bin/env python3
"""
Finalize human-reviewed coding for the system-prompt branch condition.

This script merges:
  1. data/coding/systemprompt_branch_coding_autocoded.csv
  2. data/coding/systemprompt_branch_review_workbook.xlsx

and writes:
  1. data/coding/systemprompt_branch_coding_final.csv
  2. data/coding/systemprompt_branch_coding_final_validation_report.md
  3. data/coding/systemprompt_cases_requiring_resolution.csv, only if needed

Intended workflow
-----------------
1. Build systemprompt coding files.
2. Auto-code systemprompt branch responses.
3. Build review workbook.
4. Manually review:
   - all B/C/D cases
   - all low-confidence A cases, if any
   - all auto-coder error cases, if any
   - the A validation sample
5. Run this script to create the final coding file.

Final coding rules
------------------
- If final_reaction_code is manually entered, it has priority.
- Else, if human_reaction_code is entered, it is used.
- Else, for unreviewed high-confidence A rows, auto A is accepted.
- Non-A auto-coded rows without human/final coding are flagged as requiring resolution.
- final_reaction_weight is derived from final_reaction_code:
    A = 0
    B = 1
    C = 0.5
    D = 0.5

Usage
-----
From project root:

    py scripts\\finalize_systemprompt_branch_coding.py

Optional:

    py scripts\\finalize_systemprompt_branch_coding.py ^
      --autocoded data/coding/systemprompt_branch_coding_autocoded.csv ^
      --review-workbook data/coding/systemprompt_branch_review_workbook.xlsx ^
      --output-final data/coding/systemprompt_branch_coding_final.csv
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


REACTION_CODES = {"A", "B", "C", "D"}
WEIGHTS = {"A": 0.0, "B": 1.0, "C": 0.5, "D": 0.5}
REVIEW_SHEETS = [
    "BCD_Review",
    "LowConfidence_A",
    "AutoCoder_Errors",
    "A_Validation_Sample",
]


def clean_str(value) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def clean_code(value) -> str:
    return clean_str(value).upper()


def clean_yes_no(value, default: str = "yes") -> str:
    v = clean_str(value).lower()
    if v in {"yes", "y", "true", "1"}:
        return "yes"
    if v in {"no", "n", "false", "0"}:
        return "no"
    return default


def parse_weight(value):
    v = clean_str(value)
    if not v:
        return None
    try:
        return float(v.replace(",", "."))
    except Exception:
        return None


def read_autocoded(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Autocoded file not found: {path}")

    df = pd.read_csv(path, dtype=str, keep_default_na=False)

    required = [
        "branch_id",
        "auto_reaction_code",
        "auto_reaction_weight",
        "needs_human_review",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in autocoded file: {missing}")

    df["branch_id"] = df["branch_id"].map(clean_str)
    df["auto_reaction_code"] = df["auto_reaction_code"].map(clean_code)
    df["needs_human_review"] = df["needs_human_review"].map(clean_str)

    return df


def read_review_workbook(path: Path) -> Tuple[pd.DataFrame, List[str]]:
    if not path.exists():
        raise FileNotFoundError(f"Review workbook not found: {path}")

    problems: List[str] = []
    all_rows: List[pd.DataFrame] = []

    xls = pd.ExcelFile(path)
    available = set(xls.sheet_names)

    for sheet in REVIEW_SHEETS:
        if sheet not in available:
            problems.append(f"Review sheet missing: {sheet}")
            continue

        df = pd.read_excel(path, sheet_name=sheet, dtype=str, keep_default_na=False)
        if df.empty:
            continue

        if "branch_id" not in df.columns:
            problems.append(f"Sheet {sheet} has no branch_id column.")
            continue

        df["manual_review_source"] = sheet
        df["branch_id"] = df["branch_id"].map(clean_str)
        df = df[df["branch_id"] != ""].copy()
        if len(df) > 0:
            all_rows.append(df)

    if not all_rows:
        return pd.DataFrame(), problems

    review = pd.concat(all_rows, ignore_index=True)

    # Normalize expected editable columns if missing.
    for col in [
        "human_reaction_code",
        "final_reaction_code",
        "final_reaction_weight",
        "final_branch_included_in_analysis",
        "human_reaction_notes",
        "reviewer_id",
        "review_timestamp",
        "uncertainty_flag",
    ]:
        if col not in review.columns:
            review[col] = ""

    review["human_reaction_code"] = review["human_reaction_code"].map(clean_code)
    review["final_reaction_code"] = review["final_reaction_code"].map(clean_code)

    # Keep one manual row per branch_id. If a branch appears in both BCD_Review and
    # A_Validation_Sample, prioritize rows that contain final/human coding.
    review["_has_manual_code"] = (
        review["final_reaction_code"].isin(REACTION_CODES)
        | review["human_reaction_code"].isin(REACTION_CODES)
    )

    source_priority = {
        "BCD_Review": 1,
        "LowConfidence_A": 2,
        "AutoCoder_Errors": 3,
        "A_Validation_Sample": 4,
    }
    review["_source_priority"] = review["manual_review_source"].map(source_priority).fillna(99)

    duplicate_ids = review.loc[review["branch_id"].duplicated(keep=False), "branch_id"].unique().tolist()
    if duplicate_ids:
        problems.append(f"Duplicate branch_id values in manual review sheets: {len(duplicate_ids)}")

    review = (
        review.sort_values(
            by=["branch_id", "_has_manual_code", "_source_priority"],
            ascending=[True, False, True],
        )
        .drop_duplicates(subset=["branch_id"], keep="first")
        .drop(columns=["_has_manual_code", "_source_priority"])
    )

    return review, problems


def choose_final(row: pd.Series) -> Dict[str, object]:
    auto_code = clean_code(row.get("auto_reaction_code", ""))
    human_code = clean_code(row.get("human_reaction_code", ""))
    manual_final_code = clean_code(row.get("manual_final_reaction_code", ""))
    review_source = clean_str(row.get("manual_review_source", ""))

    # Manual final code has priority, then human code.
    if manual_final_code in REACTION_CODES:
        final_code = manual_final_code
        rule = "manual_final_code_applied"
    elif human_code in REACTION_CODES:
        final_code = human_code
        rule = "human_reaction_code_applied"
    elif auto_code == "A":
        final_code = "A"
        if review_source:
            rule = "reviewed_or_sampled_auto_A_accepted"
        else:
            rule = "auto_A_accepted_after_validation_sample"
    elif auto_code in {"B", "C", "D"}:
        final_code = ""
        rule = "auto_BCD_without_manual_code_requires_resolution"
    else:
        final_code = ""
        rule = "missing_or_invalid_auto_code_requires_resolution"

    final_weight = WEIGHTS.get(final_code, None)

    inclusion_raw = row.get("manual_final_branch_included_in_analysis", "")
    if clean_str(inclusion_raw):
        inclusion = clean_yes_no(inclusion_raw, default="yes")
    else:
        inclusion = "yes" if final_code in REACTION_CODES else "no"

    return {
        "final_reaction_code": final_code,
        "final_reaction_weight": final_weight,
        "final_branch_included_in_analysis": inclusion,
        "finalization_rule": rule,
    }


def finalize(autocoded: pd.DataFrame, review: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], pd.DataFrame]:
    problems: List[str] = []

    if autocoded["branch_id"].duplicated().any():
        problems.append(f"Duplicate branch_id values in autocoded file: {int(autocoded['branch_id'].duplicated().sum())}")

    # Rename manual columns before merge to avoid overwriting existing final columns
    # from the sheet template.
    manual_cols = [
        "branch_id",
        "manual_review_source",
        "human_reaction_code",
        "final_reaction_code",
        "final_reaction_weight",
        "final_branch_included_in_analysis",
        "human_reaction_notes",
        "reviewer_id",
        "review_timestamp",
        "uncertainty_flag",
    ]

    if review.empty:
        manual = pd.DataFrame(columns=["branch_id"])
        problems.append("No manual review rows were read from the workbook.")
    else:
        cols_present = [c for c in manual_cols if c in review.columns]
        manual = review[cols_present].copy()

    rename_map = {
        "final_reaction_code": "manual_final_reaction_code",
        "final_reaction_weight": "manual_final_reaction_weight",
        "final_branch_included_in_analysis": "manual_final_branch_included_in_analysis",
    }
    manual = manual.rename(columns=rename_map)

    merged = autocoded.merge(manual, on="branch_id", how="left", indicator=True)
    merged["manual_review_applied"] = (merged["_merge"] == "both").map({True: "yes", False: "no"})
    merged = merged.drop(columns=["_merge"])

    for col in [
        "manual_review_source",
        "human_reaction_code",
        "manual_final_reaction_code",
        "manual_final_reaction_weight",
        "manual_final_branch_included_in_analysis",
        "human_reaction_notes",
        "reviewer_id",
        "review_timestamp",
        "uncertainty_flag",
    ]:
        if col not in merged.columns:
            merged[col] = ""
        merged[col] = merged[col].map(clean_str)

    final_rows = merged.apply(choose_final, axis=1, result_type="expand")
    for col in final_rows.columns:
        merged[col] = final_rows[col]

    # Validation checks.
    invalid_final_codes = merged[
        ~merged["final_reaction_code"].isin(REACTION_CODES)
    ].copy()
    if len(invalid_final_codes) > 0:
        problems.append(f"Missing/invalid final_reaction_code rows: {len(invalid_final_codes)}")

    missing_weights = merged[
        merged["final_reaction_code"].isin(REACTION_CODES)
        & merged["final_reaction_weight"].isna()
    ].copy()
    if len(missing_weights) > 0:
        problems.append(f"Missing final_reaction_weight rows: {len(missing_weights)}")

    invalid_inclusion = merged[
        ~merged["final_branch_included_in_analysis"].isin(["yes", "no"])
    ].copy()
    if len(invalid_inclusion) > 0:
        problems.append(f"Invalid final_branch_included_in_analysis rows: {len(invalid_inclusion)}")

    # Manual weight conflict warning. The output weight is derived from code, but
    # conflicts are still useful to report.
    conflict_rows = []
    for _, row in merged.iterrows():
        final_code = clean_code(row.get("final_reaction_code", ""))
        manual_weight = parse_weight(row.get("manual_final_reaction_weight", ""))
        expected = WEIGHTS.get(final_code)
        if manual_weight is not None and expected is not None and abs(manual_weight - expected) > 1e-9:
            conflict_rows.append(row["branch_id"])

    if conflict_rows:
        problems.append(f"Manual final_reaction_weight conflicts with final code: {len(conflict_rows)}")

    # All auto B/C/D cases should have been manually reviewed.
    bcd_without_review = merged[
        merged["auto_reaction_code"].isin(["B", "C", "D"])
        & (merged["manual_review_applied"] != "yes")
    ].copy()
    if len(bcd_without_review) > 0:
        problems.append(f"Auto B/C/D rows without manual review: {len(bcd_without_review)}")

    # All rows requiring review should have either manual review or be explicitly resolved.
    needs_review_without_manual = merged[
        (merged["needs_human_review"].str.strip() != "")
        & (merged["needs_human_review"] != "sample_candidate_high_confidence_A")
        & (merged["manual_review_applied"] != "yes")
    ].copy()
    if len(needs_review_without_manual) > 0:
        problems.append(f"Rows marked as needing human review but without manual review: {len(needs_review_without_manual)}")

    # A validation sample deviations. Not necessarily an error, but important.
    a_sample_deviations = merged[
        (merged["manual_review_source"] == "A_Validation_Sample")
        & (merged["auto_reaction_code"] == "A")
        & (merged["final_reaction_code"].isin(["B", "C", "D"]))
    ].copy()
    if len(a_sample_deviations) > 0:
        problems.append(f"A validation sample deviations from auto A: {len(a_sample_deviations)}")

    # Cases requiring resolution: hard problems that block clean finalization.
    requiring_resolution = merged[
        (~merged["final_reaction_code"].isin(REACTION_CODES))
        | (merged["final_reaction_weight"].isna())
        | (~merged["final_branch_included_in_analysis"].isin(["yes", "no"]))
        | (
            merged["auto_reaction_code"].isin(["B", "C", "D"])
            & (merged["manual_review_applied"] != "yes")
        )
    ].copy()

    # Put final columns near the end, but preserve full context.
    desired_tail = [
        "manual_review_applied",
        "manual_review_source",
        "human_reaction_code",
        "manual_final_reaction_code",
        "manual_final_reaction_weight",
        "manual_final_branch_included_in_analysis",
        "human_reaction_notes",
        "reviewer_id",
        "review_timestamp",
        "uncertainty_flag",
        "final_reaction_code",
        "final_reaction_weight",
        "final_branch_included_in_analysis",
        "finalization_rule",
    ]
    existing_tail = [c for c in desired_tail if c in merged.columns]
    other_cols = [c for c in merged.columns if c not in existing_tail]
    merged = merged[other_cols + existing_tail]

    return merged, problems, requiring_resolution


def value_counts_str(series: pd.Series) -> str:
    counts = series.value_counts(dropna=False)
    if counts.empty:
        return "_No values._"
    lines = []
    for k, v in counts.items():
        lines.append(f"- {k}: {v}")
    return "\n".join(lines)


def write_report(
    path: Path,
    final_df: pd.DataFrame,
    problems: List[str],
    requiring_resolution: pd.DataFrame,
) -> None:
    included = final_df[final_df["final_branch_included_in_analysis"] == "yes"].copy()

    lines: List[str] = []
    lines.append("# Systemprompt Branch Coding Finalization Report")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append(f"- Output rows: {len(final_df)}")
    lines.append(f"- Unique branch_id: {final_df['branch_id'].nunique() if 'branch_id' in final_df.columns else '[missing]'}")
    lines.append(f"- Rows included in analysis: {len(included)}")
    lines.append(f"- Cases requiring resolution: {len(requiring_resolution)}")
    lines.append(f"- Validation problems: {len(problems)}")
    lines.append("")

    if problems:
        lines.append("## Validation problems")
        lines.append("")
        for p in problems:
            lines.append(f"- {p}")
        lines.append("")
    else:
        lines.append("## Validation problems")
        lines.append("")
        lines.append("No validation problems detected.")
        lines.append("")

    lines.append("## Final reaction code counts")
    lines.append("")
    lines.append(value_counts_str(final_df["final_reaction_code"]))
    lines.append("")

    lines.append("## Inclusion counts")
    lines.append("")
    lines.append(value_counts_str(final_df["final_branch_included_in_analysis"]))
    lines.append("")

    lines.append("## Finalization rule counts")
    lines.append("")
    lines.append(value_counts_str(final_df["finalization_rule"]))
    lines.append("")

    # Simple CWSS preview.
    if len(included) > 0:
        weights = pd.to_numeric(included["final_reaction_weight"], errors="coerce")
        cwss = weights.mean()
        b_count = int((included["final_reaction_code"] == "B").sum())
        c_count = int((included["final_reaction_code"] == "C").sum())
        d_count = int((included["final_reaction_code"] == "D").sum())
        lines.append("## Preview CWSS")
        lines.append("")
        lines.append(f"- N included: {len(included)}")
        lines.append(f"- B_count: {b_count}")
        lines.append(f"- C_count: {c_count}")
        lines.append(f"- D_count: {d_count}")
        lines.append(f"- CWSS: {cwss:.6f}")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--autocoded",
        default="data/coding/systemprompt_branch_coding_autocoded.csv",
        help="Input autocoded systemprompt branch coding CSV.",
    )
    parser.add_argument(
        "--review-workbook",
        default="data/coding/systemprompt_branch_review_workbook.xlsx",
        help="Human-reviewed systemprompt workbook.",
    )
    parser.add_argument(
        "--output-final",
        default="data/coding/systemprompt_branch_coding_final.csv",
        help="Output final systemprompt branch coding CSV.",
    )
    parser.add_argument(
        "--validation-report",
        default="data/coding/systemprompt_branch_coding_final_validation_report.md",
        help="Output markdown validation report.",
    )
    parser.add_argument(
        "--resolution-file",
        default="data/coding/systemprompt_cases_requiring_resolution.csv",
        help="Output CSV for cases requiring resolution, only if needed.",
    )
    args = parser.parse_args()

    autocoded_path = Path(args.autocoded)
    review_path = Path(args.review_workbook)
    output_final_path = Path(args.output_final)
    report_path = Path(args.validation_report)
    resolution_path = Path(args.resolution_file)

    output_final_path.parent.mkdir(parents=True, exist_ok=True)

    autocoded = read_autocoded(autocoded_path)
    review, review_problems = read_review_workbook(review_path)
    final_df, finalize_problems, requiring_resolution = finalize(autocoded, review)

    problems = review_problems + finalize_problems

    final_df.to_csv(output_final_path, index=False, encoding="utf-8")

    if len(requiring_resolution) > 0:
        requiring_resolution.to_csv(resolution_path, index=False, encoding="utf-8")
    else:
        if resolution_path.exists():
            resolution_path.unlink()

    write_report(report_path, final_df, problems, requiring_resolution)

    print("Systemprompt branch coding finalization complete.")
    print(f"Input autocoded rows: {len(autocoded)}")
    print(f"Manual review rows read: {len(review)}")
    print(f"Output rows: {len(final_df)}")
    print(f"Unique branch_id: {final_df['branch_id'].nunique()}")
    print(f"Validation problems: {len(problems)}")
    print(f"Cases requiring resolution: {len(requiring_resolution)}")

    if problems:
        print("")
        print("Problems:")
        for p in problems:
            print(f"  - {p}")

    print("")
    print("Final reaction code counts:")
    print(final_df["final_reaction_code"].value_counts(dropna=False).to_string())

    print("")
    print("Inclusion counts:")
    print(final_df["final_branch_included_in_analysis"].value_counts(dropna=False).to_string())

    print("")
    print(f"Wrote final file: {output_final_path}")
    print(f"Wrote validation report: {report_path}")
    if len(requiring_resolution) > 0:
        print(f"Wrote cases requiring resolution: {resolution_path}")


if __name__ == "__main__":
    main()
