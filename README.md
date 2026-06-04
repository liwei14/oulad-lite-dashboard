# OULAD-Lite LA Dashboard

**Adaptive for teacher agency and students' SRL** — proof-of-concept for itec BAP-2026-283 (KU Leuven + imec).

## Purpose

Explores **teacher agency**, **communities of practice**, and **student data autonomy** in learning-analytics design. It is **not** a claim to research-grade ML expertise; in one interface it shows how the OULAD data are prepared, how learning signals are surfaced to teachers and students, and where ethical limits apply.

## Data source

[Open University Learning Analytics Dataset (OULAD)](https://analyse.kmi.open.ac.uk/open-dataset) (Kuzilek et al. 2017, *Scientific Data* 4:170171), CC-BY 4.0. This demo uses modules **BBB** (humanities) and **DDD** (STEM) with all behavioural and assessment features computed at **day 28** (no post-hoc leakage).

## Three special design choices

1. **Adaptive LA for teacher agency (and digital capability)** — Three collapsible info groups (What is / What's behind / What next) plus a **Hide model predictions** toggle. *Anchors: Lapage, Crabbé, & Depaepe (2026) on adaptable dashboards that support instructional autonomy; Van Leeuwen & Rummel (2020) on division of responsibility in advisory mode; Depaepe-style deliberate non-use.*

2. **Teacher community of practice** — Case conferences and a peer timeline replace cross-teacher “alert” surveillance; students must be informed and may attend. *Anchors: Tinto (1993); Wenger (1998); Slade & Prinsloo (2013); Prinsloo & Slade (2014) on intrusion, surveillance, and care. Hugo Li's doctoral research in the history of education reform likewise emphasises teacher networking and professional community as drivers of individual development.*

3. **Student data-erasure as pedagogical right** — Students see which data categories were cleared and what is retained, then start fresh without model charts or percentiles; teachers are notified and can preview model consequences in **Data erasure (educator view)** on BBB/DDD pages. *Anchors: Masschelein & Simons (2013) scholè; Simons & Masschelein (2021) on the right to begin anew free from algorithmic persona; GDPR-style data-subject rights as pedagogical principle.*

The **Student** tab shows a full-cohort rhythm summary plus **per-module (BBB / DDD) descriptive views** (engagement, TMA timing/score, class click patterns)—no risk scores or model probabilities. The default demo profile is the test-set student enrolled in **both** BBB and DDD; module sections use collapsed expanders until opened.

## How to run

```bash
cd OULAD-Lite-Dashboard
source .venv/bin/activate
python -m pytest tests/ -v
streamlit run app/Home.py
```

Symlink the seven OULAD CSVs under `data/raw/` (see project setup). Interim parquets under `data/interim/` are produced by the data-loader and feature pipeline (Steps 2–3).

**Streamlit Cloud:** use `.python-version` (3.11) and the pinned `requirements.txt`; regenerate `outputs/model_artifacts.joblib` with the same stack (e.g. `.venv311` + `train_models`) before pushing.

## Honest limitations

- **Day-28 features only** — the model never sees behaviour or assessments after week 4.
- **Clickstream** — a behavioural proxy only, not engagement quality.
- *Footnote: VLE clicks log activity, not self-regulated learning (Zimmerman 2002; Panadero 2017).*
- **No TMA text or forum post content** — humanities-facing signals rely on clickstream proxies; OULAD does not include what students wrote.
- **Single cohort, no external validation** — not tested on out-of-time or out-of-module data.
- **Single model trained jointly on BBB + DDD** — the BBB/DDD UI differentiation (thresholds, language) is methodological and informed by PCK literature, not a per-module model; a per-module model would be a natural v2 extension.

## What this demo does NOT collect

No forum post text content; no facial or camera data; no emotion or affect recognition; no behavioural data beyond the platform clickstream that OULAD itself contains. Data minimisation constrains what predictive models can plausibly say.

## What this demo does NOT claim

That OULAD’s DDD module is specifically mathematics (it is anonymised STEM); that model outputs should drive teacher decisions (they are starting points for discussion); that erasure makes a student “less at risk” (it changes the model’s information, not the learning trajectory); that institutional deployments could keep “student decides, teacher cannot block” without registrar, DPO, or ethics-committee review.

## Theoretical anchors

- Van Leeuwen & Rummel (2020) — teacher agency in LA.
- Depaepe et al. (2023), i-Learn Paper 4 — deliberate non-use of LA.
- Depaepe, *Bewust Digitaal* — ethical digital pedagogy.
- Hattie & Timperley (2007) — feedback harmful when framed as fixed failure labels.
- Tinto (1993) — academic and social integration.
- Wenger (1998) — communities of practice.
- Hargreaves & Fullan (2012) — professional capital.
- Masschelein & Simons (2013) — scholè and release from past records.
- Shulman (1986/87) PCK; Grigaliūnienė, Lehtinen, Verschaffel & Depaepe (2025, ZDM) — topic-specific PCK in mathematics.
- Hardt (2016); Kleinberg, Mullainathan & Raghavan / Chouldechova — fairness impossibility and error-rate disparities.

## Reproducibility

Before showcasing (e.g., interview or submission), run `pip freeze > requirements-lock.txt` and commit it alongside the loose-pinned `requirements.txt`. Development stays light; the lock file pins a specific environment for reproduction.
