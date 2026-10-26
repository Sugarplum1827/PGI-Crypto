# PGI-Crypto

**Prime Gap Classification and Cryptographic Key-Generation Research Prototype**

## What it is

PGI-Crypto is a research library and Streamlit presentation system for studying statistical patterns in consecutive prime gaps. It turns a window of prime gaps into numerical features, applies statistical summaries and machine-learning classifiers, and presents the resulting analyses. It also includes an experimental RSA-style key-generation demonstration that reports how the PGI classifier behaves as part of that workflow.

The classes produced by the classifier—`CC`, `MA`, `MGV`, and `NONE`—are labels defined for this study. They describe the study's heuristic categories; they are not cryptographic security ratings.

## What the system does

The analysis pipeline works with a centered window of 100 consecutive prime gaps (101 primes). For each window it can:

- calculate gap-distribution features and analytical statistics, including the study's CR ratio and an exponential-distribution goodness-of-fit statistic;
- predict one of the four study labels and report class probabilities;
- summarize model features with global or per-window SHAP explanations;
- validate prime sequences and the structure of an RSA-style result;
- display dataset summaries, classification results, model evaluation, and the Experimental demonstration in the Streamlit panel.

The panel has Data, Classification, SHAP, PGI-Crypto, and Evaluation sections. The PGI-Crypto section presents the Experimental workflow: model predictions outside the model's observed range are reported but are not enforced as a key-approval decision. Trial records do not save private-key material.

## Models included

This folder contains two distinct model tracks:

| Track | Contents and role | Data scope |
|---|---|---|
| Frozen baseline | Two archived XGBoost models retained for reproducing the earlier two-stage classifier | 24-feature baseline associated with prime midpoints through `10^8` |
| Rebuilt classifier | A separately trained four-class XGBoost model used by the research-system API and panel | 22 model features; observed sample midpoint range is 233 through 999,990,307,727 |

The rebuilt model manifest reports weighted, spatially grouped five-fold results of about 84.0% accuracy, 0.540 macro-F1, and 0.717 balanced accuracy. Its reported MGV precision is about 0.000129, so overall accuracy must not be read as strong detection of every class. The panel also distinguishes model predictions from the analytical rules used to define study labels.

## What is in this folder

- Python modules for prime-gap windows, features, analytical rules, classification, validation, and the experimental key-generation workflow.
- `models/` — frozen baseline models, the rebuilt classifier, and model provenance files.
- `panel_data/` — model and dataset manifests, demonstration records, evaluation summaries, GPU metadata, and Experimental-trial summaries used by the panel.
- `panel_static/` — static presentation assets retained with the package.
- `data/` — a small catalogue of maximal-gap records used by the study's MGV labeling rule; it is not the full prime-gap dataset.
- `streamlit_app.py` — the panel's Streamlit interface.

The full-range prime-gap stream, processed training samples, complete experiment outputs, and thesis documents belong to the separate research project; they are not bundled in this library directory.

## Research scope and limitations

The rebuilt classifier's observed training support ends below `10^12`. RSA-2048 prime factors are approximately 1024 bits long and lie far outside that support. The Experimental demonstration can generate and structurally validate RSA-style results using the operating system's random source, but its out-of-range PGI predictions do not validate those factors, prove cryptographic strength, or establish an advantage over standard key generation.

PGI-Crypto is a research prototype, not a production cryptographic system or a security certification. The included measurements describe this implementation and its study data only; they do not establish cryptographic superiority.
