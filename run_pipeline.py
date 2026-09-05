"""
Kopi Luwak Sentiment Analysis Pipeline
========================================
Consolidated from 8 Jupyter notebooks into a single executable script.

Stages:
  1. Ingest & language stratification (from 01_ingest_and_language.ipynb)
  2. Cleaning & normalization       (from 02_cleaning.ipynb)
  3. Keyword backfill               (from 03_keyword_backfill.ipynb)
  4. Dual-stream sentiment scoring  (from 04_features_and_sentiment.ipynb)
  5. Intent + stance classification (from 05_intent_and_stance.ipynb)

Output:
  outputs/final/stance_labeled_dataset.csv

Usage:
  1. Ensure the `luwak` package is installed (pip install -e .)
  2. Place raw Excel file at data/raw/kopi_luwak_5tahun_master.xlsx
  3. Run: python run_pipeline.py
"""

import sys
import logging
from pathlib import Path

# ---------------------------------------------------------------------------
# 0.  Project root & config
# ---------------------------------------------------------------------------
ROOT = next(
    p for p in [Path.cwd(), *Path.cwd().parents]
    if (p / "pyproject.toml").exists()
)
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 60)

from luwak.config import Config, configure_logging, set_global_seed

configure_logging(logging.INFO)
cfg = Config.default(ROOT)
set_global_seed(cfg.seed)

OUT_DIR = ROOT / "outputs" / "final"
OUT_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)


# ===================================================================
# Stage 1 — Ingest & language stratification
# ===================================================================
def stage_ingest_and_language():
    """Read the raw Excel workbook, deduplicate, and keep only EN/ID rows."""
    from luwak import io_utils, language, pipeline

    ledger = io_utils.Ledger()
    raw = pipeline.run_ingest(cfg, ledger)
    frame, exclusions, annotated = pipeline.run_language(cfg, raw, ledger)

    io_utils.save_stage(frame, cfg, "02_language")
    io_utils.save_stage(annotated, cfg, "02_language_annotated")
    io_utils.save_table(exclusions, cfg, "language_exclusions")
    ledger.save(cfg.paths.tables / "row_ledger.csv")

    logger.info(
        "Stage 1 complete: %d rows survived language filter (EN=%d, ID=%d)",
        len(frame),
        int((frame["lang_final"] == "en").sum()),
        int((frame["lang_final"] == "in").sum()),
    )
    return frame, ledger


# ===================================================================
# Stage 2 — Cleaning & normalization
# ===================================================================
def stage_cleaning(frame, ledger):
    """Normalize text, drop thin rows, flag verbatim repeats."""
    from luwak import cleaning, io_utils

    cleaned = cleaning.clean_frame(frame, cfg.cleaning)
    cleaned = cleaning.drop_thin_rows(cleaned, cfg.language.min_tokens)
    cleaned = cleaning.flag_duplicate_text(cleaned)

    LEDGER_PATH = cfg.paths.tables / "row_ledger.csv"
    ledger.record(
        "03_cleaned", len(cleaned), f"min_tokens={cfg.language.min_tokens}"
    )
    ledger.record(
        "03_unique_text",
        int((~cleaned["is_text_duplicate"]).sum()),
        "verbatim repeats flagged",
    )
    ledger.save(LEDGER_PATH)
    io_utils.save_stage(cleaned, cfg, "03_cleaned")

    n_unique = int((~cleaned["is_text_duplicate"]).sum())
    logger.info(
        "Stage 2 complete: %d rows, %d unique texts",
        len(cleaned),
        n_unique,
    )
    return cleaned, ledger


# ===================================================================
# Stage 3 — Keyword backfill (one keyword per row)
# ===================================================================
def stage_keyword_backfill(frame):
    """Assign one keyword per row via lexicon → self-training cascade."""
    from luwak import io_utils, keyword_model, keywords

    frame["keyword_primary"] = keywords.primary_original(frame)
    frame = keywords.apply_lexicon(frame)

    result = keyword_model.backfill_keywords(frame, cfg.keyword, cfg.seed)
    tagged = result.frame

    io_utils.save_stage(tagged, cfg, "04_keywords")
    logger.info(
        "Stage 3 complete: keyword_source distribution:\n%s",
        result.source_counts.to_string(),
    )
    return tagged


# ===================================================================
# Stage 4 — Features & dual-stream sentiment (RoBERTa + IndoBERT)
# ===================================================================
def stage_sentiment(frame):
    """Score sentiment with RoBERTa (EN) and IndoBERT (ID), align via Z-score."""
    from luwak import features, io_utils, sentiment

    frame = features.add_token_counts(frame, cfg.sentiment)
    scored, backends = sentiment.score_frame(frame, cfg.sentiment)

    io_utils.save_stage(scored, cfg, "06_scored")
    io_utils.save_metadata(
        cfg,
        "sentiment",
        {
            "backends": backends.as_dict(),
            "vader_convergence": sentiment.agreement_with_vader(scored),
            "alignment": scored.groupby("lang_final")["polarity_raw"]
            .agg(["mean", "std", "count"])
            .round(6)
            .to_dict(),
        },
    )
    logger.info(
        "Stage 4 complete: backends=%s",
        backends.as_dict(),
    )
    return scored


# ===================================================================
# Stage 5 — Intent & stance classification
# ===================================================================
def stage_intent_and_stance(frame):
    """
    Two-stage classification:
      Stage 1: intent (organic / noise / marketing)
      Stage 2: 4-class stance on organic rows (PRO / CON / AMBIVALENT / NEUTRAL)
    """
    from luwak import concepts, intent, io_utils, stance
    from luwak.id_lexicon import verify_lexicon

    frame = concepts.annotate_concepts(frame)

    # --- Stage 1: intent classification ---
    ruled = intent.apply_rules(frame, cfg.intent)
    frame, intent_history, intent_eval = intent.classify_intent(
        frame, cfg.intent, cfg.seed
    )

    # --- Stage 2: four-class stance (organic rows only) ---
    frame = stance.score_frame(frame, cfg.stance)
    frame = stance.add_analytic_flag(frame)
    frame["stance_z"] = stance.align_net_zscores(frame).round(4)

    io_utils.save_stage(frame, cfg, "08_stance")

    n_organic = int(frame["is_organic"].sum())
    n_analytic = int(frame["in_analytic_corpus"].sum())
    logger.info(
        "Stage 5 complete: organic=%d, analytic_corpus=%d",
        n_organic,
        n_analytic,
    )
    return frame


# ===================================================================
# Export — save the stance-labeled dataset as CSV
# ===================================================================
def export_dataset(frame):
    """Save the final labeled dataset as CSV for downstream analysis."""
    out_path = OUT_DIR / "stance_labeled_dataset.csv"
    frame.to_csv(out_path, index=False)
    logger.info("Saved %d rows x %d cols to %s", len(frame), len(frame.columns), out_path)
    return out_path


# ===================================================================
# Main
# ===================================================================
def main():
    logger.info("=== Kopi Luwak Sentiment Pipeline ===")

    # Stage 1
    logger.info("--- Stage 1: Ingest & language stratification ---")
    frame, ledger = stage_ingest_and_language()

    # Stage 2
    logger.info("--- Stage 2: Cleaning & normalization ---")
    frame, ledger = stage_cleaning(frame, ledger)

    # Stage 3
    logger.info("--- Stage 3: Keyword backfill ---")
    frame = stage_keyword_backfill(frame)

    # Stage 4
    logger.info("--- Stage 4: Dual-stream sentiment ---")
    frame = stage_sentiment(frame)

    # Stage 5
    logger.info("--- Stage 5: Intent & stance classification ---")
    frame = stage_intent_and_stance(frame)

    # Export
    out_path = export_dataset(frame)

    # Summary
    logger.info("=== Pipeline complete ===")
    if "stance" in frame.columns and "intent" in frame.columns:
        logger.info("Intent distribution:\n%s", frame["intent"].value_counts().to_string())
        organic = frame[frame["is_organic"] == True]
        if "stance" in organic.columns:
            logger.info(
                "Stance distribution (organic rows):\n%s",
                organic["stance"].value_counts().to_string(),
            )
    logger.info("Output: %s", out_path)


if __name__ == "__main__":
    main()
