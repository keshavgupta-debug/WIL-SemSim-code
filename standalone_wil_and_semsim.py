"""Standalone Word Information Lost (WIL) and Semantic Similarity (SemSim) Evaluator.

Extracted from Compass Evals (https://github.com/karya-inc/compass-evals)
for standalone analysis, research, and collaboration.

Zero internal dependencies on compass-evals.

Dependencies:
    pip install jiwer sentence-transformers numpy pandas

Usage:
    python standalone_wil_and_semsim.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import jiwer
import numpy as np
import pandas as pd


# =====================================================================
# 1. WORD INFORMATION LOST (WIL)
# =====================================================================

@dataclass(frozen=True)
class ErrorCounts:
    """Edit counts for one sentence or accumulated over an entire corpus.

    Attributes:
        substitutions: Words substituted by the model.
        deletions: Ground-truth reference words missing from the output.
        insertions: Extra words introduced by the model.
        ref_len: Total words in the reference text.
    """

    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0
    ref_len: int = 0

    @property
    def errors(self) -> int:
        """Total edit operations (substitutions + deletions + insertions)."""
        return self.substitutions + self.deletions + self.insertions

    @property
    def hits(self) -> int:
        """Reference words the model transcribed exactly right."""
        return self.ref_len - self.substitutions - self.deletions

    @property
    def hyp_len(self) -> int:
        """Total words in the model hypothesis."""
        return self.hits + self.substitutions + self.insertions

    @property
    def wer(self) -> float:
        """Word Error Rate (errors / ref_len)."""
        return self.errors / self.ref_len if self.ref_len else float("nan")

    def __add__(self, other: ErrorCounts) -> ErrorCounts:
        """Combine error counts across items for corpus-level aggregation."""
        return ErrorCounts(
            substitutions=self.substitutions + other.substitutions,
            deletions=self.deletions + other.deletions,
            insertions=self.insertions + other.insertions,
            ref_len=self.ref_len + other.ref_len,
        )


def count_word_errors(reference: str, hypothesis: str) -> ErrorCounts:
    """Align reference and hypothesis using jiwer and return error counts."""
    ref_clean = reference.strip()
    hyp_clean = hypothesis.strip()

    if not ref_clean and not hyp_clean:
        return ErrorCounts(0, 0, 0, 0)

    out = jiwer.process_words(ref_clean, hyp_clean)
    ref_len = len(ref_clean.split()) if ref_clean else 0

    return ErrorCounts(
        substitutions=out.substitutions,
        deletions=out.deletions,
        insertions=out.insertions,
        ref_len=ref_len,
    )


def word_info_lost_rate(counts: ErrorCounts) -> float:
    """Word Information Lost (WIL), from Morris et al. (2004).

    WIL = 1 - (Hits / Reference_Words) * (Hits / Hypothesis_Words)

    Properties:
    - Strictly bounded between 0.0 (perfect transcription) and 1.0 (total loss).
    - Unlike WER, it cannot exceed 1.0 when models hallucinate or insert words.
    - Symmetrically penalizes both omissions (deletions) and hallucinations (insertions).
    """
    if not counts.ref_len:
        return float("nan")
    if not counts.hyp_len:
        return 1.0
    return 1.0 - (counts.hits / counts.ref_len) * (counts.hits / counts.hyp_len)


# =====================================================================
# 2. SEMANTIC SIMILARITY (SemSim)
# =====================================================================

_MODEL_CACHE: dict[str, Any] = {}


def load_semsim_model(model_id: str = "sentence-transformers/LaBSE"):
    """Load and cache the sentence-transformers model."""
    if model_id not in _MODEL_CACHE:
        from sentence_transformers import SentenceTransformer

        _MODEL_CACHE[model_id] = SentenceTransformer(model_id)
    return _MODEL_CACHE[model_id]


def per_item_semsim(
    references: list[str],
    hypotheses: list[str],
    model_id: str = "sentence-transformers/LaBSE",
    batch_size: int = 32,
    empty_hypothesis_score: float = 0.0,
) -> list[float]:
    """Cosine similarity between sentence embeddings of reference and hypothesis.

    Higher is better (1.0 = identical semantic meaning, 0.0 = completely unrelated).

    Steps:
    1. If hypothesis is empty or failed, immediately assign empty_hypothesis_score (0.0).
    2. Encode valid pairs using LaBSE with normalize_embeddings=True (unit L2 vectors).
    3. Compute dot product (since ||u|| = ||v|| = 1, dot product is cosine similarity).
    """
    scores: list[float] = [0.0] * len(references)
    to_encode_refs: list[str] = []
    to_encode_hyps: list[str] = []
    indices: list[int] = []

    for i, (ref, hyp) in enumerate(zip(references, hypotheses, strict=True)):
        if not hyp.strip():
            scores[i] = empty_hypothesis_score
        else:
            to_encode_refs.append(ref.strip())
            to_encode_hyps.append(hyp.strip())
            indices.append(i)

    if not indices:
        return scores

    model = load_semsim_model(model_id)
    ref_emb = model.encode(to_encode_refs, batch_size=batch_size, normalize_embeddings=True)
    hyp_emb = model.encode(to_encode_hyps, batch_size=batch_size, normalize_embeddings=True)

    # Dot product of normalized embeddings = cosine similarity
    sims = np.sum(ref_emb * hyp_emb, axis=1)

    for idx, sim in zip(indices, sims, strict=True):
        scores[idx] = float(sim)

    return scores


# =====================================================================
# 3. HIGH-LEVEL EVALUATION PIPELINE
# =====================================================================

def evaluate_asr_pairings(
    df: pd.DataFrame,
    ref_col: str = "reference",
    hyp_col: str = "hypothesis",
    semsim_model_id: str = "sentence-transformers/LaBSE",
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Score a table of reference and hypothesis pairs for WER, WIL, and SemSim.

    Returns:
        per_item: DataFrame containing per-row WER, WIL, SemSim, and edit counts.
        corpus_summary: Dict of corpus-level aggregated scores.
    """
    refs = df[ref_col].astype(str).tolist()
    hyps = df[hyp_col].astype(str).tolist()

    # 1. Compute ErrorCounts and per-item WIL / WER
    counts_list = [count_word_errors(r, h) for r, h in zip(refs, hyps, strict=True)]
    item_wers = [c.wer for c in counts_list]
    item_wils = [word_info_lost_rate(c) for c in counts_list]

    # 2. Compute per-item SemSim
    item_semsims = per_item_semsim(refs, hyps, model_id=semsim_model_id)

    # 3. Build enriched per-item DataFrame
    per_item = df.copy()
    per_item["substitutions"] = [c.substitutions for c in counts_list]
    per_item["deletions"] = [c.deletions for c in counts_list]
    per_item["insertions"] = [c.insertions for c in counts_list]
    per_item["hits"] = [c.hits for c in counts_list]
    per_item["ref_words"] = [c.ref_len for c in counts_list]
    per_item["hyp_words"] = [c.hyp_len for c in counts_list]
    per_item["wer"] = item_wers
    per_item["wil"] = item_wils
    per_item["semsim"] = item_semsims

    # 4. Corpus-level aggregation (exactly as in Compass Evals)
    # WER and WIL sum error counts across the whole corpus:
    total_counts = ErrorCounts()
    for c in counts_list:
        total_counts = total_counts + c

    corpus_wer = total_counts.wer
    corpus_wil = word_info_lost_rate(total_counts)
    corpus_semsim = float(np.mean(item_semsims))

    corpus_summary = {
        "n_samples": len(df),
        "total_ref_words": total_counts.ref_len,
        "total_hyp_words": total_counts.hyp_len,
        "corpus_wer": corpus_wer,
        "corpus_wil": corpus_wil,
        "corpus_semsim": corpus_semsim,
    }

    return per_item, corpus_summary


# =====================================================================
# 4. MOCK DATA & VERIFICATION TEST RUN
# =====================================================================

if __name__ == "__main__":
    print("=" * 95)
    print("STANDALONE WIL & SEMSIM EVALUATION TEST RUN")
    print("=" * 95)

    # Diverse mock evaluation dataset covering key ASR phenomena:
    # - Exact matches
    # - Synonym / Paraphrasing (High WER, High SemSim)
    # - Code-mixing / Sheng dialect variations
    # - Omissions / Partial deletions
    # - Hallucinations / High insertions
    # - Complete API error / Empty transcript
    mock_data = pd.DataFrame(
        [
            {
                "item_id": "item_001",
                "description": "Exact match",
                "reference": "the quick brown fox jumps over the lazy dog",
                "hypothesis": "the quick brown fox jumps over the lazy dog",
            },
            {
                "item_id": "item_002",
                "description": "Paraphrase / Synonym (Meaning preserved)",
                "reference": "i need to purchase some groceries from the supermarket",
                "hypothesis": "i have to buy some groceries from the store",
            },
            {
                "item_id": "item_003",
                "description": "Sheng / Swahili dialect substitution",
                "reference": "niko poa sana msee nakucheki baadaye",
                "hypothesis": "niko sawa sana man nitakuona baadaye",
            },
            {
                "item_id": "item_004",
                "description": "Partial omission (Deletions)",
                "reference": "the flight to nairobi departs at seven in the morning",
                "hypothesis": "the flight departs at seven",
            },
            {
                "item_id": "item_005",
                "description": "Hallucination / Extra words (Insertions)",
                "reference": "please confirm your account details",
                "hypothesis": "please confirm your personal online banking account details immediately",
            },
            {
                "item_id": "item_006",
                "description": "Empty output / API failure",
                "reference": "welcome to the meeting everyone",
                "hypothesis": "",
            },
        ]
    )

    print("\n[INFO] Running evaluation on 6 mock test cases using LaBSE...")
    per_item_results, summary = evaluate_asr_pairings(mock_data)

    print("\n" + "-" * 95)
    print(
        f"{'Item ID':<9} | {'Description':<26} | {'WER (↓)':<8} | {'WIL (↓)':<8} | {'SemSim (↑)':<10}"
    )
    print("-" * 95)
    for _, row in per_item_results.iterrows():
        print(
            f"{row['item_id']:<9} | "
            f"{row['description'][:26]:<26} | "
            f"{row['wer']:<8.3f} | "
            f"{row['wil']:<8.3f} | "
            f"{row['semsim']:<10.3f}"
        )

    print("-" * 95)
    print("\n[CORPUS-LEVEL SUMMARY]")
    print(f"  • Total samples:      {summary['n_samples']}")
    print(f"  • Total ref words:    {summary['total_ref_words']}")
    print(f"  • Total hyp words:    {summary['total_hyp_words']}")
    print(f"  • Corpus WER:         {summary['corpus_wer']:.3f} ({summary['corpus_wer'] * 100:.1f}%)")
    print(f"  • Corpus WIL:         {summary['corpus_wil']:.3f} ({summary['corpus_wil'] * 100:.1f}%)")
    print(f"  • Corpus SemSim:      {summary['corpus_semsim']:.3f}")
    print("=" * 95)
