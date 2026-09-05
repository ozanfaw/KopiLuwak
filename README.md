# KopiLuwak
Dataset collected from social media posts in the topic of civet coffee (kopi luwak)

# Kopi Luwak Sentiment Analysis Pipeline

End-to-end NLP pipeline that ingests raw posts about kopi luwak (civet coffee) and produces a stance-labeled dataset ready for downstream analysis.

# BATCH A KEYWORDS
keywords_batch_a = [
    'kopi luwak', 'civet coffee', 'luwak coffee', 'cat poop coffee',
    'civet cat coffee', 'asian palm civet', 'wild luwak',
    'wild civet coffee', 'caged civet', 'kopi luwak liar', 'luwak liar', 'luwak kandang',
    '#kopiluwak', '#civetcoffee', '#luxurycoffee', '#animalwelfare', '#asianpalmcivet'
]

Consolidated from 8 Jupyter notebooks into a single runnable script.

## What the pipeline does

```
Raw Excel (7,762 rows)
  → deduplicate on tweet ID
  → keep only English + Indonesian tweets
  → normalize text (Bahasa Alay mapped, URLs/mentions stripped)
  → assign one keyword per row (lexicon → self-training classifier)
  → score sentiment (RoBERTa for EN, IndoBERT for ID)
  → classify intent (organic / noise / marketing)
  → classify stance (PRO / CON / AMBIVALENT / NEUTRAL)
  → save final labeled dataset as CSV
```

**Expected output after a full run:** ~2,857 organic rows with 4-class stance labels.

## Requirements

- Python 3.11+
- The `luwak` package (the project source package at `D:\ScrapingLuwak`)

### Python dependencies

```
pandas
numpy
scikit-learn
transformers          # for RoBERTa / IndoBERT
torch
joblib
langdetect
openpyxl
```

```

**GPU acceleration**
→ The transformer models automatically use CUDA if available. For CPU-only runs, expect ~15–20 minutes for sentiment scoring on ~4,000 rows.
