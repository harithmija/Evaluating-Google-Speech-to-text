#!/usr/bin/env python3
"""
Quran STT evaluation pipeline (FYP: Surah Al-Mulk)

Steps (matches lecturer's list + proposal Ch.3):
  1. Qari audio, verse by verse      -> done by you (30 clips)
  2. Speech-to-text                  -> paste the online converter's output into a CSV
  3. NLP                             -> normalise -> tokenise -> Levenshtein word alignment -> S/D/I
  4. Accuracy                        -> WER, CER, Accuracy per verse and overall

Usage:
  python eval_stt.py data/transcripts/alafasy.csv --qari "Mishary Alafasy"
  python eval_stt.py data/transcripts/alafasy.txt          # one verse per line (30 lines)

Output (in results/): <qari>_verses.csv, <qari>_errors.csv, <qari>_report.html
"""
import argparse, csv, html, os, re, sys, json

HERE = os.path.dirname(os.path.abspath(__file__))
REF_CSV = os.path.join(HERE, "data", "al_mulk_reference.csv")

# ---------------------------------------------------------------- 1. NORMALISATION
# Reference is Uthmani script; online STT returns normal (Imla'i) Arabic spelling.
# Normalisation maps both to one comparable "skeleton" form.
_TATWEEL_HAMZA = re.compile("\u0640[\u064B-\u0652]*\u0654")           # tatweel + hamza above -> hamza
_MARKS = re.compile("[\u0610-\u061A\u064B-\u065F\u06D6-\u06DC\u06DF-\u06E8\u06EA-\u06ED\u0640\u08D3-\u08FF]")
_NON_ARABIC = re.compile("[^\u0621-\u063A\u0641-\u064A\\s]")           # keep Arabic letters + space
_ALEF_MEDIAL = re.compile("(?<=\\S)\u0627")                              # alef not at word start
# Uthmani writes some long-a words with waw + dagger alef (الحيوة = الحياة, الصلوة, الزكوة ...)
_UTHMANI_WAW = re.compile("(حي|صل|زك|نج|غد|مشك)و(?=\u0647$)")


def normalize(text: str, strict_alef: bool = False) -> str:
    t = text.replace("\u0670", "" if not strict_alef else "\u0627")      # dagger alef
    t = _TATWEEL_HAMZA.sub("\u0621", t)
    t = _MARKS.sub("", t)
    t = (t.replace("\u0671", "\u0627")                                     # alef wasla
          .replace("\u0623", "\u0627").replace("\u0625", "\u0627").replace("\u0622", "\u0627")
          .replace("\u0624", "\u0621").replace("\u0626", "\u0621")         # hamza carriers -> hamza
          .replace("\u0649", "\u064A").replace("\u06CC", "\u064A")         # alef maqsura / farsi yeh -> yeh
          .replace("\u0629", "\u0647")                                     # ta marbuta -> ha
          .replace("\u06A9", "\u0643"))
    t = _NON_ARABIC.sub(" ", t)                                           # punctuation, digits, latin
    t = re.sub(r"\s+", " ", t).strip()
    if not strict_alef:
        # Uthmani drops many long-vowel alefs (e.g. تَبَٰرَكَ) that normal spelling writes.
        # Removing non-initial alefs on BOTH sides avoids counting spelling convention as ASR error.
        t = " ".join(_UTHMANI_WAW.sub(r"\1", _ALEF_MEDIAL.sub("", w)) for w in t.split(" "))
    return t


def tokenize(text: str):
    return text.split() if text else []


# ---------------------------------------------------------------- 2. LEVENSHTEIN ALIGNMENT
def align(ref, hyp):
    """Word-level Levenshtein with backtrace. Returns list of (op, ref_word, hyp_word)
    where op in {'C' correct, 'S' substitution, 'D' deletion, 'I' insertion}."""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1): d[i][0] = i
    for j in range(m + 1): d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
    out, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (0 if ref[i - 1] == hyp[j - 1] else 1):
            out.append(("C" if ref[i - 1] == hyp[j - 1] else "S", ref[i - 1], hyp[j - 1])); i -= 1; j -= 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            out.append(("D", ref[i - 1], "")); i -= 1
        else:
            out.append(("I", "", hyp[j - 1])); j -= 1
    return out[::-1]


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


# ---------------------------------------------------------------- 3. METRICS
def metrics(ref_norm: str, hyp_norm: str):
    r, h = tokenize(ref_norm), tokenize(hyp_norm)
    ops = align(r, h)
    S = sum(o == "S" for o, _, _ in ops); D = sum(o == "D" for o, _, _ in ops)
    I = sum(o == "I" for o, _, _ in ops); C = sum(o == "C" for o, _, _ in ops)
    N = len(r)
    wer = (S + D + I) / N if N else 0.0
    cer = edit_distance(ref_norm, hyp_norm) / len(ref_norm) if ref_norm else 0.0
    acc = C / N * 100 if N else 0.0            # Accuracy = correct words / reference words
    return dict(N=N, C=C, S=S, D=D, I=I, WER=wer * 100, CER=cer * 100, Accuracy=acc,
                ref_chars=len(ref_norm), char_edits=edit_distance(ref_norm, hyp_norm)), ops


# ---------------------------------------------------------------- IO
def load_reference():
    with open(REF_CSV, encoding="utf-8") as f:
        return {int(r["verse"]): r["text"] for r in csv.DictReader(f)}


def load_hyp(path):
    """CSV with columns verse,stt_text  OR  txt with one verse per line (line 1 = verse 1)."""
    hyp = {}
    if path.lower().endswith(".csv"):
        with open(path, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                k = r.get("verse") or r.get("Verse")
                v = r.get("stt_text") or r.get("text") or ""
                if k and str(k).strip().isdigit():
                    hyp[int(k)] = v
    else:
        with open(path, encoding="utf-8-sig") as f:
            lines = [l.rstrip("\n") for l in f]
        for i, l in enumerate(lines, 1):
            hyp[i] = l
    return hyp


def render_alignment_html(ops):
    cells = []
    for op, r, h in ops:
        cls = {"C": "ok", "S": "sub", "D": "del", "I": "ins"}[op]
        if op == "C": txt = html.escape(r)
        elif op == "S": txt = f"<s>{html.escape(r)}</s> → {html.escape(h)}"
        elif op == "D": txt = f"<s>{html.escape(r)}</s> (missing)"
        else: txt = f"+ {html.escape(h)}"
        cells.append(f'<span class="w {cls}">{txt}</span>')
    return " ".join(cells)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hyp_file")
    ap.add_argument("--qari", default=None)
    ap.add_argument("--strict-alef", action="store_true", help="keep medial alefs (stricter, penalises Uthmani/Imla'i spelling gaps)")
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    a = ap.parse_args()

    qari = a.qari or os.path.splitext(os.path.basename(a.hyp_file))[0]
    slug = re.sub(r"[^\w]+", "_", qari).strip("_")
    os.makedirs(a.out, exist_ok=True)
    ref = load_reference(); hyp = load_hyp(a.hyp_file)

    rows, err_rows, blocks = [], [], []
    tot = dict(N=0, C=0, S=0, D=0, I=0, ref_chars=0, char_edits=0)
    missing = [v for v in ref if not (hyp.get(v) or "").strip()]
    for v in sorted(ref):
        if v in missing:        # no STT text yet -> not evaluated (not counted as deletions)
            continue
        rn = normalize(ref[v], a.strict_alef); hn = normalize(hyp.get(v, ""), a.strict_alef)
        m, ops = metrics(rn, hn)
        for k in tot: tot[k] += m[k]
        rows.append(dict(verse=v, reference=ref[v], stt_raw=hyp.get(v, ""), ref_norm=rn, stt_norm=hn,
                         **{k: (round(x, 2) if isinstance(x, float) else x) for k, x in m.items()}))
        for op, r, h in ops:
            if op != "C": err_rows.append(dict(verse=v, type={"S": "substitution", "D": "deletion", "I": "insertion"}[op], reference_word=r, stt_word=h))
        blocks.append((v, m, ops, hyp.get(v, "")))

    N = tot["N"]
    if not N: sys.exit("No STT text found - check your file format.")
    overall = dict(verses_evaluated=len(rows), N=N, C=tot["C"], S=tot["S"], D=tot["D"], I=tot["I"],
                   WER=round((tot["S"] + tot["D"] + tot["I"]) / N * 100, 2),
                   CER=round(tot["char_edits"] / tot["ref_chars"] * 100, 2),
                   Accuracy=round(tot["C"] / N * 100, 2))

    with open(os.path.join(a.out, f"{slug}_verses.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    with open(os.path.join(a.out, f"{slug}_errors.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["verse", "type", "reference_word", "stt_word"]); w.writeheader(); w.writerows(err_rows)
    with open(os.path.join(a.out, f"{slug}_summary.json"), "w", encoding="utf-8") as f:
        json.dump(dict(qari=qari, overall=overall, empty_verses=missing), f, ensure_ascii=False, indent=2)

    # HTML report
    trs = "".join(
        f"<tr><td>{v}</td><td>{m['N']}</td><td>{m['S']}</td><td>{m['D']}</td><td>{m['I']}</td>"
        f"<td>{m['WER']:.1f}</td><td>{m['CER']:.1f}</td><td>{m['Accuracy']:.1f}</td></tr>"
        for v, m, _, _ in blocks)
    det = "".join(
        f'<div class="v"><div class="vh">Verse {v} &nbsp;·&nbsp; WER {m["WER"]:.1f}% · CER {m["CER"]:.1f}% · Acc {m["Accuracy"]:.1f}%</div>'
        f'<div class="al" dir="rtl">{render_alignment_html(ops) if raw.strip() else "<i>no STT text</i>"}</div></div>'
        for v, m, ops, raw in blocks)
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>STT Evaluation – {html.escape(qari)}</title><style>
:root{{--bg:#fff;--fg:#1c1c1c;--mut:#666;--line:#ddd;--ok:#e8f5e9;--sub:#fff3cd;--del:#fde0e0;--ins:#e0ecff}}
@media(prefers-color-scheme:dark){{:root{{--bg:#16181a;--fg:#eee;--mut:#9aa;--line:#333;--ok:#1d3a22;--sub:#4a3d12;--del:#4a1f1f;--ins:#1b2f52}}}}
body{{font-family:system-ui,sans-serif;background:var(--bg);color:var(--fg);max-width:960px;margin:0 auto;padding:16px}}
h1{{font-size:1.3rem}} .kpi{{display:flex;gap:12px;flex-wrap:wrap;margin:12px 0}}
.k{{border:1px solid var(--line);border-radius:8px;padding:10px 16px}} .k b{{display:block;font-size:1.5rem}}
table{{border-collapse:collapse;width:100%;font-size:.9rem}} td,th{{border-bottom:1px solid var(--line);padding:4px 8px;text-align:right}}
.al{{font-family:'Amiri','Traditional Arabic',serif;font-size:1.35rem;line-height:2.2}} .vh{{color:var(--mut);font-size:.85rem;margin-top:14px}}
.w{{padding:1px 6px;border-radius:5px;display:inline-block}} .ok{{background:var(--ok)}} .sub{{background:var(--sub)}} .del{{background:var(--del)}} .ins{{background:var(--ins)}}
.legend span{{margin-right:8px}}</style></head><body>
<h1>Google-style STT evaluation · Surah Al-Mulk · {html.escape(qari)}</h1>
<div class="kpi"><div class="k">WER<b>{overall['WER']}%</b></div><div class="k">CER<b>{overall['CER']}%</b></div><div class="k">Accuracy<b>{overall['Accuracy']}%</b></div>
<div class="k">Verses<b>{len(rows)}/30</b></div><div class="k">Words<b>{N}</b></div><div class="k">S / D / I<b>{overall['S']} / {overall['D']} / {overall['I']}</b></div></div>
<p class="legend"><span class="w ok">correct</span><span class="w sub">substitution</span><span class="w del">deletion</span><span class="w ins">insertion</span></p>
<table><tr><th>Verse</th><th>Words</th><th>S</th><th>D</th><th>I</th><th>WER %</th><th>CER %</th><th>Acc %</th></tr>{trs}</table>
<h2>Verse-by-verse alignment</h2>{det}</body></html>"""
    with open(os.path.join(a.out, f"{slug}_report.html"), "w", encoding="utf-8") as f: f.write(page)

    print(f"Qari: {qari}")
    print(f"Verses evaluated {len(rows)}/30 | Words {N} | S {tot['S']} D {tot['D']} I {tot['I']} | WER {overall['WER']}%  CER {overall['CER']}%  Accuracy {overall['Accuracy']}%")
    if missing: print("WARNING empty STT text for verses:", missing)
    print("Results in", a.out)


if __name__ == "__main__":
    main()
