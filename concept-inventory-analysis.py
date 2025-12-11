"""
Concept Inventory Pre/Post Analysis Tool
=======================================

This script ingests two CSV files containing student responses to pre‑tests and
post‑tests for one of three concept inventories:

* EMCS   – Electric & Magnetic Concept Survey
* BEMA   – Brief Electricity & Magnetism Assessment
* EBAPS  – Epistemological Beliefs Assessment for Physical Science

For each inventory a keyed answer list and the position of the built‑in
attention‑check question are hard‑coded below.

The program grades each submission, validates attention‑check compliance,
performs fuzzy matching on the self‑reported *unique identifier* to pair pre
and post attempts, and produces summary statistics including normalized gain.

Usage (CLI)  >>>  python concept_inventory_analysis.py pre.csv post.csv --test BEMA

Dependencies
------------
* pandas
* numpy
"""
from __future__ import annotations

import argparse
import difflib
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

############################
# Hard‑coded Test Meta‑Data #
############################

TEST_KEYS: Dict[str, Dict[str, List[str] | int | str]] = {
    "EMCS": {
        "key": [
            "1", "5", "2", "1", "4", "3", "5", "3", "1", "4", "5", "4",
            "3", "4", "1", "3", "2", "5", "2", "1", "3", "4", "2", "1",
            "1", "5"
        ],
        "attention_idx": 17,  # zero‑based index of attention‑check question
        "attention_answer": "5",
    },
    "BEMA": {
        "key": [
            "C", "D", "A", "B", "C", "D", "A", "B", "D", "C",
            "A", "B", "C", "D", "A", "C", "B", "D", "A", "B",
            "C", "D", "A", "B"  
        ],
        "attention_idx": 12,
        "attention_answer": "C",
    },
    "EBAPS": {
        "key": [
            "A", "B", "B", "C", "D", "A", "C", "D", "B", "A",
            "C", "D", "B", "A", "C", "B", "D", "A", "C", "B",
        ],
        "attention_idx": 4,
        "attention_answer": "D",
    },
}

ID_COL = "identifier"  # column in CSV that stores the self‑chosen ID

##################################
# Core analysis helper functions  #
##################################

def grade_responses(df: pd.DataFrame, key: List[str]) -> pd.Series:
    """Return a Series of integer scores for each student submission."""
    # assume responses are in columns Q1, Q2, ... matching key order
    question_cols = [f"Q{i+1}" for i in range(len(key))]
    comparison = df[question_cols] == key
    return comparison.sum(axis=1)


def attention_pass(df: pd.DataFrame, idx: int, correct: str) -> pd.Series:
    """Boolean Series indicating which students passed the attention check."""
    col = f"Q{idx + 1}"
    return df[col] == correct


def best_match(id_str: str, candidates: List[str], threshold: float = 0.8) -> Tuple[str | None, float]:
    """Return the candidate with highest similarity above *threshold* (ratio)."""
    if not id_str or not candidates:
        return None, 0.0
    similarities = [(cand, difflib.SequenceMatcher(None, id_str, cand).ratio()) for cand in candidates]
    best_cand, best_score = max(similarities, key=lambda x: x[1])
    return (best_cand, best_score) if best_score >= threshold else (None, best_score)


def pair_submissions(pre: pd.DataFrame, post: pd.DataFrame, threshold: float = 0.8) -> List[Tuple[int, int]]:
    """Return list of index pairs (pre_idx, post_idx) for matched identifiers."""
    post_ids = post[ID_COL].astype(str).tolist()
    used_post_idx = set()
    pairs: List[Tuple[int, int]] = []

    for pre_idx, pre_id in pre[ID_COL].astype(str).items():
        match, score = best_match(pre_id, [pid for i, pid in enumerate(post_ids) if i not in used_post_idx], threshold)
        if match is not None:
            post_idx = post_ids.index(match)
            pairs.append((pre_idx, post_idx))
            used_post_idx.add(post_idx)
    return pairs


def normalized_gain(pre_score: int, post_score: int, max_score: int) -> float | None:
    """Calculate Hake's normalized gain. Return None if undefined."""
    if pre_score == max_score:
        return None  # already perfect; gain undefined
    return (post_score - pre_score) / (max_score - pre_score)


@dataclass
class SummaryStatistics:
    test_name: str
    n_pre: int
    n_post: int
    n_matched: int
    class_pre_mean: float
    class_post_mean: float
    class_gain_mean: float

    def __str__(self) -> str:
        return (
            f"\nSummary for {self.test_name}\n" + "-" * (12 + len(self.test_name)) +
            f"\nPre‑test respondents           : {self.n_pre}" +
            f"\nPost‑test respondents          : {self.n_post}" +
            f"\nMatched identifier pairs       : {self.n_matched}" +
            f"\nClass mean score (pre)         : {self.class_pre_mean:.2f}" +
            f"\nClass mean score (post)        : {self.class_post_mean:.2f}" +
            f"\nMean normalized gain (matched) : {self.class_gain_mean:.3f}\n"
        )


#################
# Main pipeline #
#################

def analyze(pre_csv: Path, post_csv: Path, test_name: str, id_threshold: float = 0.8) -> SummaryStatistics:
    meta = TEST_KEYS[test_name]
    key = meta["key"]
    att_idx = meta["attention_idx"]
    att_ans = meta["attention_answer"]

    pre_df = pd.read_csv(pre_csv)
    post_df = pd.read_csv(post_csv)

    # grade and filter attention check
    pre_df["score"] = grade_responses(pre_df, key)
    post_df["score"] = grade_responses(post_df, key)

    pre_df = pre_df[attention_pass(pre_df, att_idx, att_ans)]
    post_df = post_df[attention_pass(post_df, att_idx, att_ans)]

    # match identifiers with fuzzy matching
    pairs = pair_submissions(pre_df, post_df, threshold=id_threshold)

    gains = []
    for pre_idx, post_idx in pairs:
        pre_s = int(pre_df.loc[pre_idx, "score"])
        post_s = int(post_df.loc[post_idx, "score"])
        g = normalized_gain(pre_s, post_s, max_score=len(key))
        if g is not None:
            gains.append(g)

    summary = SummaryStatistics(
        test_name=test_name,
        n_pre=len(pre_df),
        n_post=len(post_df),
        n_matched=len(pairs),
        class_pre_mean=pre_df["score"].mean(),
        class_post_mean=post_df["score"].mean(),
        class_gain_mean=float(np.mean(gains)) if gains else float("nan"),
    )
    return summary


########################
# Command‑line wrapper #
########################

def _cli() -> None:
    parser = argparse.ArgumentParser(description="Analyze pre/post concept inventory data")
    parser.add_argument("pre", type=Path, help="CSV file with pre‑test responses")
    parser.add_argument("post", type=Path, help="CSV file with post‑test responses")
    parser.add_argument("--test", choices=TEST_KEYS.keys(), required=True, help="Which inventory (EMCS, BEMA, EBAPS)")
    parser.add_argument("--threshold", type=float, default=0.8, help="Fuzzy ID match similarity threshold [0‑1]")
    args = parser.parse_args()

    summary = analyze(args.pre, args.post, args.test, args.threshold)
    print(summary)


if __name__ == "__main__":
    _cli()
