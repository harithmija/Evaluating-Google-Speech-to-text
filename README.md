# Quran STT evaluation – Surah Al-Mulk (FYP)

Workflow (your lecturer's 4 steps)
1. **1 qari, verse by verse** – get 30 audio clips (one per ayah). Per-ayah mp3s of many qaris are on everyayah.com (file names like `067001.mp3`).
2. **Speech-to-text** – upload/play each clip into an online STT (Google-based), paste the Arabic output into `data/transcripts/<qari>.csv` (copy `TEMPLATE_qari.csv`, fill `stt_text`). Later this step is replaced by the Google STT API – nothing else changes.
3. **NLP** – `eval_stt.py` normalises (diacritics, alef/hamza/ya/ta-marbuta variants, Uthmani→normal spelling), tokenises, aligns with Levenshtein (word level) and classifies Substitution / Deletion / Insertion.
4. **Accuracy** – WER, CER, Accuracy per verse and overall.

Run:  `python eval_stt.py data/transcripts/<qari>.csv --qari "Qari Name"`
Needs only Python 3 (jiwer is used only to cross-check).

Outputs in `results/`: `*_verses.csv` (per-verse metrics), `*_errors.csv` (every S/D/I word), `*_summary.json`, `*_report.html` (colour-coded alignment – good for screenshots in your report).

Definitions
- WER = (S+D+I)/N × 100 (N = reference words)
- CER = char-level edit distance / reference chars × 100
- Accuracy = correct words / N × 100
- Verses with empty `stt_text` are skipped (not counted as deletions).

Notes / limits
- Default normalisation drops non-initial alefs on both sides so Uthmani vs normal spelling isn't scored as an ASR error. This means a missing/extra long-alef is not detected; use `--strict-alef` to keep them.
- Reference text: Uthmani script, 30 verses (bismillah not included), from the semarketir/quranjson dataset – double-check against your mushaf/Tanzil before final submission.
