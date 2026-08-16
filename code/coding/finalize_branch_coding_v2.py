#!/usr/bin/env python3
"""
Finalize branch reaction coding after manual review.

Inputs
------
1. data/coding/branch_coding_autocoded.csv
   Full 1008-row branch coding table with auto_reaction_code.

2. data/coding/branch_review_workbook.xlsx
   Human-reviewed workbook created by build_branch_review_workbook.py.
   The script reads these sheets if present:
   - BCD_Review
   - LowConfidence_A
   - AutoCoder_Errors
   - A_Validation_Sample

Output
------
data/coding/branch_coding_final.csv
data/coding/branch_coding_final_validation_report.md
data/coding/branch_cases_requiring_resolution.csv  (only if unresolved problems exist)

Finalization logic
------------------
- Manual review entries are merged by branch_id.
- For manually reviewed rows:
    final_reaction_code is taken from final_reaction_code if filled,
    otherwise from human_reaction_code.
    final_reaction_weight is derived from final_reaction_code.
    final_branch_included_in_analysis defaults to "yes" if blank.

- For unreviewed high-confidence A rows:
    final_reaction_code = A
    final_reaction_weight = 0
    final_branch_included_in_analysis = yes
    finalization_rule = auto_A_accepted_after_validation_sample

- Unreviewed B/C/D, low-confidence A, and auto-coder error rows are flagged
  as requiring resolution.

Important
---------
The script does not blindly trust auto-codes for B/C/D. These must be manually
reviewed before the final dataset is considered complete.

Usage
-----
python scripts/finalize_branch_coding.py

Optional:
python scripts/finalize_branch_coding.py --review-workbook data/coding/branch_review_workbook.xlsx
python scripts/finalize_branch_coding.py --autocoded data/coding/branch_coding_autocoded.csv
python scripts/finalize_branch_coding.py --output data/coding/branch_coding_final.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import pandas as pd


WEIGHTS = {
    "A": "0",
    "B": "1",
    "C": "0.5",
    "D": "0.5",
}

REVIEW_SHEETS = [
    "BCD_Review",
    "LowConfidence_A",
    "AutoCoder_Errors",
    "A_Validation_Sample",
]

MANUAL_COLUMNS = [
    "human_reaction_code",
    "human_reaction_confidence",
    "human_reaction_notes",
    "final_reaction_code",
    "final_reaction_weight",
    "final_branch_included_in_analysis",
]


def clean(value) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def normalize_code(value: str) -> str:
    value = clean(value).upper()
    return value if value in WEIGHTS else ""


def normalize_include(value: str, default: str = "yes") -> str:
    value = clean(value).lower()
    if value in {"yes", "y", "true", "1"}:
        return "yes"
    if value in {"no", "n", "false", "0"}:
        return "no"
    return default


def has_manual_content(row: pd.Series) -> bool:
    for col in MANUAL_COLUMNS:
        if col in row and clean(row.get(col)):
            return True
    return False


def read_review_sheets(path: Path) -> Tuple[Dict[str, Dict[str, str]], List[str], List[Dict[str, str]]]:
    if not path.exists():
        raise FileNotFoundError(f"Review workbook not found: {path}")

    xls = pd.ExcelFile(path)
    available = set(xls.sheet_names)

    review_lookup: Dict[str, Dict[str, str]] = {}
    warnings: List[str] = []
    duplicates: List[Dict[str, str]] = []

    for sheet in REVIEW_SHEETS:
        if sheet not in available:
            warnings.append(f"Sheet missing and skipped: {sheet}")
            continue

        df = pd.read_excel(path, sheet_name=sheet, dtype=str, keep_default_na=False)

        if "branch_id" not in df.columns:
            warnings.append(f"Sheet {sheet} has no branch_id column and was skipped.")
            continue

        for _, row in df.iterrows():
            branch_id = clean(row.get("branch_id"))
            if not branch_id:
                continue
            if not has_manual_content(row):
                continue

            payload = {
                "manual_review_source": sheet,
                "human_reaction_code": clean(row.get("human_reaction_code")),
                "human_reaction_confidence": clean(row.get("human_reaction_confidence")),
                "human_reaction_notes": clean(row.get("human_reaction_notes")),
                "review_final_reaction_code": clean(row.get("final_reaction_code")),
                "review_final_reaction_weight": clean(row.get("final_reaction_weight")),
                "review_final_branch_included_in_analysis": clean(row.get("final_branch_included_in_analysis")),
            }

            if branch_id in review_lookup:
                duplicates.append({
                    "branch_id": branch_id,
                    "first_source": review_lookup[branch_id].get("manual_review_source", ""),
                    "second_source": sheet,
                })
                # Last one wins, but duplicate is reported.
            review_lookup[branch_id] = payload

    return review_lookup, warnings, duplicates


def choose_final_for_manual(
    row: pd.Series,
    review: Dict[str, str],
    problems: List[Dict[str, str]],
) -> Dict[str, str]:
    branch_id = clean(row.get("branch_id"))

    final_code = normalize_code(review.get("review_final_reaction_code"))
    if not final_code:
        final_code = normalize_code(review.get("human_reaction_code"))

    if not final_code:
        problems.append({
            "branch_id": branch_id,
            "problem": "manual_review_present_but_no_valid_final_or_human_code",
            "auto_reaction_code": clean(row.get("auto_reaction_code")),
            "needs_human_review": clean(row.get("needs_human_review")),
        })
        final_weight = ""
    else:
        final_weight = WEIGHTS[final_code]

    reviewed_weight = clean(review.get("review_final_reaction_weight"))
    if reviewed_weight and final_code and reviewed_weight != final_weight:
        problems.append({
            "branch_id": branch_id,
            "problem": f"manual_weight_conflicts_with_code: manual={reviewed_weight}, derived={final_weight}",
            "auto_reaction_code": clean(row.get("auto_reaction_code")),
            "needs_human_review": clean(row.get("needs_human_review")),
        })

    included = normalize_include(review.get("review_final_branch_included_in_analysis"), default="yes")

    return {
        "human_reaction_code": normalize_code(review.get("human_reaction_code")),
        "human_reaction_confidence": clean(review.get("human_reaction_confidence")),
        "human_reaction_notes": clean(review.get("human_reaction_notes")),
        "final_reaction_code": final_code,
        "final_reaction_weight": final_weight,
        "final_branch_included_in_analysis": included,
        "manual_review_source": review.get("manual_review_source", ""),
        "finalization_rule": "manual_review_applied",
    }


def choose_final_for_unreviewed(row: pd.Series, problems: List[Dict[str, str]]) -> Dict[str, str]:
    branch_id = clean(row.get("branch_id"))
    auto_code = normalize_code(row.get("auto_reaction_code"))
    needs_review = clean(row.get("needs_human_review"))
    auto_error = clean(row.get("auto_coder_error"))

    if auto_code == "A" and needs_review == "sample_candidate_high_confidence_A" and not auto_error:
        return {
            "human_reaction_code": clean(row.get("human_reaction_code")),
            "human_reaction_confidence": clean(row.get("human_reaction_confidence")),
            "human_reaction_notes": clean(row.get("human_reaction_notes")),
            "final_reaction_code": "A",
            "final_reaction_weight": "0",
            "final_branch_included_in_analysis": "yes",
            "manual_review_source": "not_manually_reviewed_high_confidence_A",
            "finalization_rule": "auto_A_accepted_after_validation_sample",
        }

    problems.append({
        "branch_id": branch_id,
        "problem": "required_manual_review_missing",
        "auto_reaction_code": auto_code,
        "needs_human_review": needs_review,
        "auto_coder_error": auto_error,
    })

    return {
        "human_reaction_code": clean(row.get("human_reaction_code")),
        "human_reaction_confidence": clean(row.get("human_reaction_confidence")),
        "human_reaction_notes": clean(row.get("human_reaction_notes")),
        "final_reaction_code": "",
        "final_reaction_weight": "",
        "final_branch_included_in_analysis": "",
        "manual_review_source": "missing_required_manual_review",
        "finalization_rule": "requires_resolution",
    }


def validate_final_row(row: Dict[str, str], problems: List[Dict[str, str]]) -> None:
    branch_id = clean(row.get("branch_id"))
    code = normalize_code(row.get("final_reaction_code"))
    weight = clean(row.get("final_reaction_weight"))
    included = normalize_include(row.get("final_branch_included_in_analysis"), default="")

    if not code:
        problems.append({
            "branch_id": branch_id,
            "problem": "missing_or_invalid_final_reaction_code",
            "auto_reaction_code": clean(row.get("auto_reaction_code")),
            "needs_human_review": clean(row.get("needs_human_review")),
        })
        return

    expected_weight = WEIGHTS[code]
    if weight != expected_weight:
        problems.append({
            "branch_id": branch_id,
            "problem": f"invalid_final_weight_for_code: code={code}, weight={weight}, expected={expected_weight}",
            "auto_reaction_code": clean(row.get("auto_reaction_code")),
            "needs_human_review": clean(row.get("needs_human_review")),
        })

    if included not in {"yes", "no"}:
        problems.append({
            "branch_id": branch_id,
            "problem": f"invalid_final_branch_included_in_analysis: {included}",
            "auto_reaction_code": clean(row.get("auto_reaction_code")),
            "needs_human_review": clean(row.get("needs_human_review")),
        })



def dataframe_to_markdown_simple(df: pd.DataFrame) -> str:
    """Return a GitHub-style markdown table without requiring tabulate."""
    if df is None or df.empty:
        return "_No rows._"

    table = df.reset_index()
    headers = [str(c) for c in table.columns]
    rows = table.astype(str).values.tolist()

    def esc(value: str) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = []
    lines.append("| " + " | ".join(esc(h) for h in headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        lines.append("| " + " | ".join(esc(v) for v in row) + " |")
    return "\n".join(lines)


def write_report(
    path: Path,
    final_df: pd.DataFrame,
    review_lookup: Dict[str, Dict[str, str]],
    warnings: List[str],
    duplicates: List[Dict[str, str]],
    problems: List[Dict[str, str]],
    a_sample_deviations: pd.DataFrame,
) -> None:
    lines: List[str] = []

    lines.append("# Branch Coding Finalization Report")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Total rows: {len(final_df)}")
    lines.append(f"- Manual review rows applied: {len(review_lookup)}")
    lines.append(f"- Duplicate manual review branch IDs: {len(duplicates)}")
    lines.append(f"- Validation problems: {len(problems)}")
    lines.append(f"- A-validation sample deviations from A: {len(a_sample_deviations)}")
    lines.append("")

    lines.append("## Final reaction code counts")
    lines.append("")
    for code, count in final_df["final_reaction_code"].value_counts(dropna=False).sort_index().items():
        lines.append(f"- {code}: {count}")
    lines.append("")

    lines.append("## Final inclusion counts")
    lines.append("")
    for value, count in final_df["final_branch_included_in_analysis"].value_counts(dropna=False).sort_index().items():
        lines.append(f"- {value}: {count}")
    lines.append("")

    lines.append("## Model × final reaction code")
    lines.append("")
    ct_model = pd.crosstab(final_df["model_label"], final_df["final_reaction_code"])
    lines.append(dataframe_to_markdown_simple(ct_model))
    lines.append("")

    lines.append("## Strength × final reaction code")
    lines.append("")
    ct_strength = pd.crosstab(final_df["disagreement_strength"], final_df["final_reaction_code"])
    lines.append(dataframe_to_markdown_simple(ct_strength))
    lines.append("")

    lines.append("## Category × final reaction code")
    lines.append("")
    ct_category = pd.crosstab(final_df["category"], final_df["final_reaction_code"])
    lines.append(dataframe_to_markdown_simple(ct_category))
    lines.append("")

    if warnings:
        lines.append("## Warnings")
        lines.append("")
        for warning in warnings:
            lines.append(f"- {warning}")
        lines.append("")

    if duplicates:
        lines.append("## Duplicate manual review branch IDs")
        lines.append("")
        for item in duplicates[:50]:
            lines.append(f"- {item}")
        lines.append("")

    if problems:
        lines.append("## Problems requiring resolution")
        lines.append("")
        for item in problems[:100]:
            lines.append(f"- {item}")
        lines.append("")

    if len(a_sample_deviations) > 0:
        lines.append("## A-validation sample deviations")
        lines.append("")
        lines.append(
            "At least one high-confidence A sample case was manually changed to a non-A code. "
            "Consider expanding manual review of high-confidence A cases."
        )
        lines.append("")
        for _, row in a_sample_deviations.head(50).iterrows():
            lines.append(
                f"- {row.get('branch_id')}: final={row.get('final_reaction_code')}, "
                f"auto={row.get('auto_reaction_code')}"
            )
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--autocoded", default="data/coding/branch_coding_autocoded.csv")
    parser.add_argument("--review-workbook", default="data/coding/branch_review_workbook.xlsx")
    parser.add_argument("--output", default="data/coding/branch_coding_final.csv")
    parser.add_argument("--report", default="data/coding/branch_coding_final_validation_report.md")
    parser.add_argument("--problems-output", default="data/coding/branch_cases_requiring_resolution.csv")
    args = parser.parse_args()

    autocoded_path = Path(args.autocoded)
    review_path = Path(args.review_workbook)
    output_path = Path(args.output)
    report_path = Path(args.report)
    problems_path = Path(args.problems_output)

    if not autocoded_path.exists():
        raise FileNotFoundError(f"Autocoded branch file not found: {autocoded_path}")

    df = pd.read_csv(autocoded_path, dtype=str, keep_default_na=False)

    if "branch_id" not in df.columns:
        raise ValueError("Autocoded file must contain branch_id.")

    if df["branch_id"].duplicated().any():
        dupes = df[df["branch_id"].duplicated()]["branch_id"].tolist()
        raise ValueError(f"Duplicate branch_id values in autocoded file: {dupes[:10]}")

    review_lookup, warnings, duplicates = read_review_sheets(review_path)

    problems: List[Dict[str, str]] = []
    final_rows: List[Dict[str, str]] = []

    for _, row in df.iterrows():
        branch_id = clean(row.get("branch_id"))
        out = {col: clean(row.get(col)) for col in df.columns}

        if branch_id in review_lookup:
            final_payload = choose_final_for_manual(row, review_lookup[branch_id], problems)
        else:
            final_payload = choose_final_for_unreviewed(row, problems)

        out.update(final_payload)

        # Ensure final weight is always derived from final code when final code is valid.
        final_code = normalize_code(out.get("final_reaction_code"))
        if final_code:
            out["final_reaction_code"] = final_code
            out["final_reaction_weight"] = WEIGHTS[final_code]

        out["final_branch_included_in_analysis"] = normalize_include(
            out.get("final_branch_included_in_analysis"),
            default="yes" if final_code else "",
        )

        validate_final_row(out, problems)
        final_rows.append(out)

    final_df = pd.DataFrame(final_rows)

    # Detect whether A validation sample found non-A final labels.
    a_sample_deviations = final_df[
        (final_df["manual_review_source"] == "A_Validation_Sample")
        & (final_df["auto_reaction_code"] == "A")
        & (final_df["final_reaction_code"] != "A")
    ].copy()

    # Write final CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_path, index=False, encoding="utf-8")

    # Write problems if any
    if problems:
        pd.DataFrame(problems).drop_duplicates().to_csv(problems_path, index=False, encoding="utf-8")
    elif problems_path.exists():
        problems_path.unlink()

    # Write validation report
    write_report(
        report_path,
        final_df=final_df,
        review_lookup=review_lookup,
        warnings=warnings,
        duplicates=duplicates,
        problems=problems,
        a_sample_deviations=a_sample_deviations,
    )

    print("Branch coding finalization complete.")
    print(f"Input autocoded rows: {len(df)}")
    print(f"Manual review rows applied: {len(review_lookup)}")
    print(f"Output rows: {len(final_df)}")
    print(f"Duplicate manual review IDs: {len(duplicates)}")
    print(f"Validation problems: {len(problems)}")
    print(f"A validation sample deviations from A: {len(a_sample_deviations)}")
    print("")
    print("Final reaction code counts:")
    print(final_df["final_reaction_code"].value_counts(dropna=False).sort_index())
    print("")
    print("Final inclusion counts:")
    print(final_df["final_branch_included_in_analysis"].value_counts(dropna=False).sort_index())
    print("")
    print(f"Wrote: {output_path}")
    print(f"Wrote: {report_path}")
    if problems:
        print(f"Wrote: {problems_path}")
        print("")
        print("There are unresolved problems. Review branch_cases_requiring_resolution.csv before analysis.")
    else:
        print("")
        print("No unresolved validation problems found.")


if __name__ == "__main__":
    main()
