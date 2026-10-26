# Frozen key-generation baseline provenance

This package is copied into the recreated system as the supplied PGI-Crypto key-generation baseline. It is not the model newly trained by `pgicrypto.train_compare` and its scores must not be presented as results of the rebuilt training run.

- Original source: `extracted/pgi_crypto_project/pgi_crypto_project/pgi_crypto_core/pgi_crypto`.
- Frozen model bundle: Stage 1 binary XGBoost and Stage 2 three-class XGBoost, each trained on 24 features from prime data bounded by `p_mid <= 10^8`.
- The model files are hash-checked by `config.py` at load time.
- The package's strict key-generation policy rejects candidate prime sizes outside that validated interval. The experimental policy does not turn extrapolated predictions into validated cryptographic evidence.
- The source archive documents 0/12 strict key-generation successes in its 2048-bit evidence and MGV recall of 0/12 in the frozen end-to-end evaluation. Those are archived results, not newly rerun measurements.

The new data and evaluation pipeline lives in `../pgicrypto`. This baseline remains available for reproducing the original key-generation path and for measuring the scale-gate failure honestly.

## Rebuilt classifier adapter

`rebuilt_classifier.py` is an additive runtime adapter for the separately trained four-class XGBoost model in the project-level `models/` directory. It selects the clean 22-feature schema from the library's 24-column feature matrix. It does not replace the frozen two-stage model or bypass its key-generation policy. The rebuilt classifier covers a range-spanning sample through `p_mid <= 10^12`; 1024-bit cryptographic prime candidates remain far outside that range.
