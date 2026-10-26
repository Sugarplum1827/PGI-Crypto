"""Streamlit presentation system for the PGI-Crypto thesis panel."""
from __future__ import annotations

import io
import importlib.util
import json
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

APP_PATH = Path(__file__).resolve()
PACKAGE_DIR = APP_PATH.parent
PROJECT_ROOT = APP_PATH.parents[3]
SOURCE_ROOT = PROJECT_ROOT / "src" / "python"
if not SOURCE_ROOT.is_dir():
    # Streamlit Cloud checks out this repository with the app and package
    # modules at the repository root, rather than under src/python.
    PROJECT_ROOT = PACKAGE_DIR
    SOURCE_ROOT = PACKAGE_DIR.parent
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

# The flat checkout is itself a Python package, but its directory name is not
# a valid import name. Give it the expected package name so relative imports
# continue to work in Streamlit Cloud.
if importlib.util.find_spec("pgi_crypto") is None and (PACKAGE_DIR / "__init__.py").is_file():
    package_spec = importlib.util.spec_from_file_location(
        "pgi_crypto",
        PACKAGE_DIR / "__init__.py",
        submodule_search_locations=[str(PACKAGE_DIR)],
    )
    package_module = importlib.util.module_from_spec(package_spec)
    sys.modules["pgi_crypto"] = package_module

from pgi_crypto.config import ML_FEATURE_NAMES, RNG_MODE_SECURE  # noqa: E402
from pgi_crypto.eligibility import POLICY_EXPERIMENTAL  # noqa: E402
from pgi_crypto.exceptions import KeyGenerationError  # noqa: E402
from pgi_crypto.features import compute_features  # noqa: E402
from pgi_crypto.key_generation import PGICryptoKeyGenerator  # noqa: E402

MODEL_FEATURE_NAMES = tuple(name for name in ML_FEATURE_NAMES if name not in {
    "poisson_lambda_fit", "CR_log2_legacy_DO_NOT_USE_FOR_CLASSIFICATION",
})
CLASS_NAMES = ("CC", "MA", "MGV", "NONE")
MODEL_SAMPLE_PATH = PROJECT_ROOT / "data" / "processed" / "windows_1e12_model_sample10000.parquet"
FEATURE_SAMPLE_PATH = PROJECT_ROOT / "data" / "processed" / "windows_1e12_sample2000.parquet"
MODEL_DIR = PROJECT_ROOT / "models"
PANEL_DATA_DIR = PACKAGE_DIR / "panel_data"

st.set_page_config(
    page_title="PGI-Crypto | Prime Gap Classification & Cryptographic Key Generation",
    page_icon="∿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
      .block-container {max-width: 1440px; padding: 1.25rem 2rem 3rem 2rem;}
      [data-testid="stAppViewContainer"], [data-testid="stHeader"] {background:#0b1220;}
      [data-testid="stHeader"] {border-bottom:1px solid #263349;}
      h1,h2,h3 {color:#e8eef8!important; letter-spacing:-.02em;}
      h1 {font-size:1.45rem!important; font-weight:750!important; text-align:center; margin:0 0 .1rem 0!important;}
      .app-subtitle {text-align:center;color:#9aa9bf;font-size:.82rem;margin:0 0 1rem 0;}
      [role="tablist"] {justify-content:center!important; gap:1.35rem; border-bottom:1px solid #263349; width:100%;}
      [role="tab"] {height:2.7rem!important; padding:0 .8rem!important; color:#9aa9bf!important; font-size:.82rem!important;}
      [role="tab"][aria-selected="true"] {color:#e8eef8!important; font-weight:700!important;}
      [data-testid="stMetric"] {background:#121d2e;border:1px solid #29364a;border-radius:6px;padding:.8rem .95rem;}
      [data-testid="stMetricLabel"] {color:#9aa9bf;font-size:.72rem;}
      [data-testid="stMetricValue"] {color:#e8eef8;font-size:1.35rem;}
      [data-testid="stVerticalBlockBorderWrapper"] {border-color:#29364a!important;border-radius:7px!important;}
      div[data-testid="stFileUploader"] section {background:#101a2b;border:1px dashed #42516a;border-radius:6px;}
      .result-box {border:1px solid #42516a;border-radius:12px;width:130px;height:108px;margin:10px auto 6px;display:flex;align-items:center;justify-content:center;font-size:1.8rem;font-weight:600;color:#e8eef8;background:#121d2e;}
      .center-caption {text-align:center;color:#9aa9bf;font-size:.8rem;margin:.15rem 0;}
      .small-muted {font-size:.75rem;color:#9aa9bf;}
      .good-note {padding:.75rem .9rem;border:1px solid #285447;background:#102a26;border-radius:6px;color:#a7dfc7;font-size:.82rem;}
      .stDownloadButton button {border:1px solid #34445c;background:#172438;color:#e2eaf5;font-size:.75rem;}
      footer {visibility:hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_parquet(path_text: str, modified: float) -> pd.DataFrame:
    del modified  # included in the cache key so changed files reload
    return pd.read_parquet(path_text)


@st.cache_data(show_spinner=False)
def load_json_file(path_text: str, modified: float) -> dict:
    del modified
    return json.loads(Path(path_text).read_text(encoding="utf-8"))


def json_record(name: str, project_relative: str | None = None) -> dict:
    candidates = []
    if project_relative:
        candidates.append(PROJECT_ROOT / project_relative)
    candidates.append(PANEL_DATA_DIR / name)
    for path in candidates:
        if path.is_file():
            return load_json_file(str(path), path.stat().st_mtime)
    return {}


def normalize_experimental_progress(progress: dict) -> dict:
    """Normalize live checkpoints and the completed benchmark summary for the UI."""
    if "completed_trials" in progress:
        return progress
    record = next((item for item in progress.get("summaries_completed", [])
                   if item.get("policy") == "experimental_rule_gate"), None)
    if record is None:
        return {
            "policy": "experimental_rule_gate",
            "requested_trials": int(progress.get("requested_trials", 1000)),
            "completed_trials": 0,
            "successful_key_pairs": 0,
            "statuses": {},
            "mean_seconds_completed": 0.0,
            "private_key_material_persisted": False,
        }
    statuses = dict(record.get("statuses", {}))
    return {
        "policy": record.get("policy"),
        "requested_trials": int(record.get("requested_trials", 1000)),
        "completed_trials": sum(int(value) for value in statuses.values()),
        "successful_key_pairs": int(record.get("successful_key_pairs", 0)),
        "statuses": statuses,
        "blocked_before_candidate_generation": int(record.get("blocked_before_candidate_generation", 0)),
        "mean_seconds_completed": float(record.get("mean_request_or_generation_seconds") or 0.0),
        "median_seconds_completed": record.get("median_request_or_generation_seconds"),
        "p95_seconds_completed": record.get("p95_request_or_generation_seconds"),
        "model_sha256": record.get("model_sha256"),
        "model_track": record.get("model_track"),
        "private_key_material_persisted": False,
    }


def parquet_frame(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    return load_parquet(str(path), path.stat().st_mtime)


def show_main_header() -> None:
    st.markdown("<h1>PGI-CRYPTO</h1>", unsafe_allow_html=True)
    st.markdown(
        '<div class="app-subtitle">Prime Gap Classification &amp; Cryptographic Key Generation</div>',
        unsafe_allow_html=True,
    )


def summarize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    preferred = [
        "window_id", "mean_gap", "std_dev", "max_gap", "gap_entropy",
        "label_adviser_classical_4class", "sample_weight",
    ]
    available = [name for name in preferred if name in df.columns]
    return df.loc[:, available] if available else df.iloc[:, :10]


def read_uploaded_table(uploaded) -> pd.DataFrame:
    data = uploaded.getvalue()
    if uploaded.name.lower().endswith(".parquet"):
        return pd.read_parquet(io.BytesIO(data))
    return pd.read_csv(io.BytesIO(data))


def extract_uploaded_gap_features(uploaded, start_prime: int, window_size: int, step_size: int) -> tuple[pd.DataFrame, int]:
    raw = pd.read_csv(io.BytesIO(uploaded.getvalue()))
    if raw.empty:
        raise ValueError("the uploaded CSV contains no rows")
    column = "gap" if "gap" in raw.columns else raw.columns[0]
    numeric = pd.to_numeric(raw[column], errors="coerce")
    if numeric.isna().any() or not np.equal(numeric, np.floor(numeric)).all():
        raise ValueError("the gap column must contain only integer values")
    gaps = numeric.to_numpy(dtype=np.int64)
    if len(gaps) < window_size:
        raise ValueError(f"at least {window_size} gaps are required")
    if (gaps < 1).any() or (gaps > 65535).any():
        raise ValueError("gap values must be in the range 1 through 65535")
    starts = np.arange(0, len(gaps) - window_size + 1, step_size, dtype=np.int64)
    total_windows = len(starts)
    # Keep the interactive operation bounded while retaining coverage over the uploaded range.
    if total_windows > 10_000:
        starts = np.unique(np.linspace(0, total_windows - 1, 10_000, dtype=np.int64))
    matrices = np.lib.stride_tricks.sliding_window_view(gaps, window_size)[starts]
    prime_at_gap = int(start_prime) + np.concatenate(([0], np.cumsum(gaps, dtype=np.int64)))
    p_start = prime_at_gap[starts]
    p_mid = prime_at_gap[starts + window_size // 2]
    p_end = prime_at_gap[starts + window_size]
    features = compute_features(matrices, p_mid)
    frame = pd.DataFrame(features.values, columns=ML_FEATURE_NAMES)
    frame.insert(0, "window_id", starts)
    frame.insert(1, "p_start", p_start)
    frame.insert(2, "p_mid", p_mid)
    frame.insert(3, "p_end", p_end)
    return frame, total_windows


def make_trend(df: pd.DataFrame) -> pd.DataFrame | None:
    if not {"p_mid", "mean_gap"}.issubset(df.columns):
        return None
    ordered = df[["p_mid", "mean_gap"]].dropna().sort_values("p_mid")
    if len(ordered) < 2:
        return None
    ordered = ordered.iloc[::max(1, len(ordered) // 100_000)]
    bins = min(10, max(2, len(ordered) // 20))
    try:
        ordered = ordered.assign(range_band=pd.qcut(ordered["p_mid"], q=bins, duplicates="drop"))
        trend = ordered.groupby("range_band", observed=True)["mean_gap"].mean().reset_index()
        trend["range_band"] = [f"R{i + 1:02d}" for i in range(len(trend))]
        return trend.set_index("range_band")
    except (ValueError, TypeError):
        return None


def make_heatmap(df: pd.DataFrame) -> alt.Chart | None:
    selected = [
        name for name in ("mean_gap", "std_dev", "max_gap", "skewness", "gap_entropy", "twin_prime_density")
        if name in df.columns
    ]
    if len(selected) < 2:
        return None
    sample = df[selected].dropna()
    if len(sample) > 20_000:
        sample = sample.sample(20_000, random_state=20261026)
    corr = sample.corr(numeric_only=True)
    long = corr.rename_axis("feature_y").reset_index().melt(id_vars="feature_y", var_name="feature_x", value_name="correlation")
    return alt.Chart(long).mark_rect().encode(
        x=alt.X("feature_x:N", sort=selected, title=None),
        y=alt.Y("feature_y:N", sort=selected, title=None),
        color=alt.Color("correlation:Q", scale=alt.Scale(domain=[-1, 1], scheme="blues"), title="Correlation"),
        tooltip=["feature_x", "feature_y", alt.Tooltip("correlation:Q", format=".3f")],
    ).properties(height=240)


def model_summary_records() -> tuple[dict, list[dict]]:
    summary = json_record("model_comparison_summary.json", "results/metrics/full_1e12/model_comparison_summary.json")
    return summary, summary.get("summary", [])


def crypto_records() -> tuple[dict, list[dict]]:
    summary = json_record("crypto_comparison.json", "results/benchmarks/full_1e12_crypto_comparison.json")
    return summary, summary.get("results", [])


def track_name(value: str) -> str:
    return {
        "label_adviser_classical_4class": "Adviser / classical",
        "label_audited_4class": "Audited p-value",
        "label_thesis_3class": "Thesis literal (not trainable on this sample)",
    }.get(value, value)


def nice_model_name(value: str) -> str:
    return value.replace("SoftVotingEnsemble_uncalibrated", "Soft voting · uncalibrated")


def performance_frame(records: list[dict], track: str) -> pd.DataFrame:
    rows = [record for record in records if record.get("method") == track]
    output = []
    for record in rows:
        mgv = record.get("per_class", {}).get("MGV", {})
        output.append({
            "Model": nice_model_name(record.get("model", "")),
            "Accuracy": record.get("accuracy_mean"),
            "Macro-F1": record.get("macro_f1_mean"),
            "Balanced accuracy": record.get("balanced_accuracy_mean"),
            "Log loss": record.get("log_loss_mean"),
            "MGV precision": mgv.get("precision"),
            "MGV recall": mgv.get("recall"),
            "MGV mean fold support": mgv.get("support"),
        })
    return pd.DataFrame(output)


def class_label(value) -> str:
    if isinstance(value, str) and value in CLASS_NAMES:
        return value
    try:
        return CLASS_NAMES[int(value)]
    except (ValueError, TypeError, IndexError):
        return str(value)


@st.cache_resource(show_spinner="Loading the saved final models…")
def load_final_models(root_text: str) -> tuple[dict, dict]:
    root = Path(root_text)
    models: dict[str, object] = {}
    errors: dict[str, str] = {}
    try:
        import joblib
        path = root / "models" / "final_adviser_classical_4class_randomforest.joblib"
        if path.is_file():
            models["Random Forest"] = joblib.load(path)
        else:
            errors["Random Forest"] = "saved artifact is not available in this project copy"
        path = root / "models" / "final_adviser_classical_4class_mlp.joblib"
        if path.is_file():
            models["Multilayer Perceptron"] = joblib.load(path)
        else:
            errors["Multilayer Perceptron"] = "saved artifact is not available in this project copy"
    except Exception as exc:
        errors["CPU model artifacts"] = f"could not load ({type(exc).__name__})"
    try:
        import xgboost as xgb
        path = root / "models" / "final_adviser_classical_4class_xgboost.json"
        if not path.is_file():
            path = PACKAGE_DIR / "models" / "rebuilt_adviser_classical_4class_xgboost.json"
        model = xgb.XGBClassifier()
        model.load_model(str(path))
        models["XGBoost"] = model
    except Exception as exc:
        errors["XGBoost"] = f"could not load ({type(exc).__name__})"
    return models, errors


def predict_one(model, features: pd.DataFrame) -> tuple[str, np.ndarray]:
    if hasattr(model, "feature_names_in_"):
        model_input = features
    elif model.__class__.__module__.startswith(("sklearn.", "imblearn.")):
        model_input = features.to_numpy(dtype=np.float32)
    else:
        model_input = features
    raw_probabilities = np.asarray(model.predict_proba(model_input))[0]
    classes = list(getattr(model, "classes_", range(len(raw_probabilities))))
    aligned = np.zeros(len(CLASS_NAMES), dtype=np.float64)
    for raw_class, probability in zip(classes, raw_probabilities):
        label = class_label(raw_class)
        if label in CLASS_NAMES:
            aligned[CLASS_NAMES.index(label)] = float(probability)
    raw_prediction = model.predict(model_input)[0]
    return class_label(raw_prediction), aligned


def selected_evaluation_row(df: pd.DataFrame, window_id: int) -> pd.Series | None:
    if "window_id" not in df.columns:
        return None
    matched = df.loc[df["window_id"].astype("int64") == int(window_id)]
    return None if matched.empty else matched.iloc[0]


def draw_data_tab(default_feature_df: pd.DataFrame | None, dataset_meta: dict) -> None:
    left, right = st.columns([0.92, 1.85], gap="large")
    with left:
        with st.container(border=True):
            st.subheader("Prime Gap Dataset")
            st.caption("Load a processed feature table or inspect the bundled full-range sample.")
            uploaded_table = st.file_uploader("Upload feature CSV or Parquet", type=["csv", "parquet"], key="feature_table_upload")
            if uploaded_table is not None:
                try:
                    st.session_state["uploaded_table"] = read_uploaded_table(uploaded_table)
                    st.success(f"Loaded {uploaded_table.name}")
                except Exception as exc:
                    st.error(f"Could not read the uploaded table: {type(exc).__name__}")
            row_count = len(st.session_state.get("uploaded_table", default_feature_df)) if st.session_state.get("uploaded_table", default_feature_df) is not None else 0
            m1, m2 = st.columns(2)
            m1.metric("Records", f"{row_count:,}" if row_count else "—")
            m2.metric("Model inputs", "22")
            st.caption("24 stored audit features; 22 predictors in the rebuilt models.")

        with st.container(border=True):
            st.subheader("Feature Engineering")
            a, b = st.columns(2)
            window_size = a.number_input("Window size (gaps)", min_value=6, max_value=500, value=100, step=1)
            step_size = b.number_input("Step size (gaps)", min_value=1, max_value=500, value=50, step=1)
            first_prime = st.number_input("First prime in uploaded gap stream", min_value=2, value=2, step=1)
            gap_upload = st.file_uploader("Upload a CSV with a 'gap' column", type=["csv"], key="raw_gap_upload")
            if st.button("Extract Features", width="stretch", type="primary", key="extract_features_button"):
                if gap_upload is None:
                    st.info("Upload a small raw gap CSV to extract new features. The study samples below are already processed.")
                else:
                    try:
                        extracted, total = extract_uploaded_gap_features(gap_upload, int(first_prime), int(window_size), int(step_size))
                        st.session_state["extracted_feature_table"] = extracted
                        st.session_state["extracted_total_windows"] = total
                        st.success(f"Extracted {len(extracted):,} windows from {total:,} possible windows.")
                    except Exception as exc:
                        st.error(f"Feature extraction stopped: {exc}")

    if default_feature_df is None:
        with right:
            st.warning("The bundled Parquet sample was not found. Add the project data folder or upload a processed feature table.")
        return

    current = st.session_state.get("uploaded_table")
    extracted = st.session_state.get("extracted_feature_table")
    df = current if isinstance(current, pd.DataFrame) else (extracted if isinstance(extracted, pd.DataFrame) else default_feature_df)
    with right:
        with st.container(border=True):
            heading, controls = st.columns([1.6, 1])
            with heading:
                st.subheader("Dataset Preview")
                source_label = "Uploaded table" if current is not None else ("Extracted windows" if extracted is not None else "Full-range feature sample")
                st.caption(f"{source_label} · Showing up to 10,000 rows")
            with controls:
                full_view = st.toggle("Full view", value=False, key="full_feature_view")
                export_data = st.download_button(
                    "Export JSON",
                    data=summarize_dataframe(df).head(10_000).to_json(orient="records", indent=2),
                    file_name="pgi_crypto_feature_preview.json",
                    mime="application/json",
                    width="stretch",
                )
                del export_data
            preview = df.head(10_000 if full_view else 20)
            st.dataframe(preview if full_view else summarize_dataframe(preview), width="stretch", height=310, hide_index=True)

        chart_left, chart_right = st.columns(2, gap="large")
        with chart_left:
            with st.container(border=True):
                st.subheader("Gap Distribution Trend")
                trend = make_trend(df)
                if trend is not None:
                    st.area_chart(trend, color="#596575", height=240, width="stretch")
                    st.caption("Mean gap by ordered midpoint-prime range in the sampled data.")
                else:
                    st.info("This table needs p_mid and mean_gap columns to draw the trend.")
        with chart_right:
            with st.container(border=True):
                st.subheader("Feature Correlation Heatmap")
                heatmap = make_heatmap(df)
                if heatmap is not None:
                    st.altair_chart(heatmap, width="stretch")
                else:
                    st.info("Upload or select a feature table with at least two numeric feature columns.")
    if dataset_meta:
        st.caption(
            f"Source study: {dataset_meta.get('gap_count', 0):,} generated gaps through the inclusive 10¹² limit; "
            "the displayed feature table is a weighted sample, not exhaustive all-window extraction."
        )


def draw_classification_tab(eval_df: pd.DataFrame | None, summary: dict, demo: dict) -> None:
    left, right = st.columns([0.92, 1.85], gap="large")
    model_track = "label_adviser_classical_4class"
    selected_models = ["Random Forest", "XGBoost", "Multilayer Perceptron"]
    selected_row = None
    selected_id = int(demo.get("window_id", 0))
    selected_row_index = 0
    if eval_df is not None:
        matching_demo = eval_df.index[eval_df["window_id"].astype("int64") == selected_id].tolist()
        default_index = int(matching_demo[0]) if matching_demo else 0
        selected_row_index = int(st.session_state.get("selected_row_index", default_index))

    with left:
        with st.container(border=True):
            st.subheader("PRIME-GAP CLASSIFICATION")
            if eval_df is None:
                st.warning("The model-evaluation Parquet sample is not available.")
            else:
                c1, c2 = st.columns(2)
                c1.metric("Input rows", f"{len(eval_df):,}")
                c2.metric("Features", "22")
                selected_row_index = int(st.slider(
                    "Sample row",
                    min_value=0,
                    max_value=len(eval_df) - 1,
                    value=min(max(selected_row_index, 0), len(eval_df) - 1),
                    help="Choose an observed row from the saved model-evaluation sample.",
                ))
                st.session_state["selected_row_index"] = selected_row_index
                selected_row = eval_df.iloc[selected_row_index]
                selected_id = int(selected_row["window_id"])
                st.session_state["selected_window_id"] = selected_id
                st.caption(f"Window {selected_id:,} · p_mid = {int(selected_row['p_mid']):,} · adviser/classical target = {selected_row.get('label_adviser_classical_4class', 'not recorded')}")
                available = ["Random Forest", "XGBoost", "Multilayer Perceptron"]
                selected_models = st.multiselect("Models", available, default=available)
                st.caption("Run Classification performs final-model inference on the selected row. It is not an independent test prediction.")
                run = st.button("Run Classification", type="primary", width="stretch", disabled=selected_row is None or not selected_models)
            if eval_df is None:
                run = False
        if run and selected_row is not None:
            models, errors = load_final_models(str(PROJECT_ROOT))
            missing = [name for name in selected_models if name not in models]
            if missing:
                st.error("Unavailable saved models: " + ", ".join(missing) + ". " + "; ".join(errors.values()))
            else:
                try:
                    x = pd.DataFrame([selected_row[list(MODEL_FEATURE_NAMES)].astype(float).to_dict()], columns=MODEL_FEATURE_NAMES)
                    predictions = {}
                    aligned_probabilities = []
                    for name in selected_models:
                        label, probabilities = predict_one(models[name], x)
                        predictions[name] = {"label": label, "probabilities": probabilities.tolist()}
                        aligned_probabilities.append(probabilities)
                    ensemble = np.mean(np.vstack(aligned_probabilities), axis=0)
                    predictions["Final average ensemble"] = {
                        "label": CLASS_NAMES[int(np.argmax(ensemble))],
                        "probabilities": ensemble.tolist(),
                    }
                    st.session_state["classification_result"] = {
                        "window_id": selected_id,
                        "actual_label": str(selected_row["label_adviser_classical_4class"]),
                        "predictions": predictions,
                        "track": "label_adviser_classical_4class",
                    }
                except Exception as exc:
                    st.error(f"Model inference could not complete: {type(exc).__name__}: {exc}")

    with right:
        with st.container(border=True):
            st.subheader("CLASSIFICATION RESULT")
            saved = st.session_state.get("classification_result")
            if saved and saved["window_id"] == selected_id:
                result = saved["predictions"]["Final average ensemble"]
                label = result["label"]
                probability = float(result["probabilities"][CLASS_NAMES.index(label)])
                st.caption(f"Selected window: {selected_id:,} · Target label: {saved['actual_label']}")
                st.markdown("<div class='center-caption'>Predicted Regime</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='result-box'>{label}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='center-caption'>Final average probability: <b>{probability:.1%}</b> · uncalibrated</div>", unsafe_allow_html=True)
                output_rows = []
                for name, record in saved["predictions"].items():
                    label_index = CLASS_NAMES.index(record["label"])
                    output_rows.append({"Model": name, "Prediction": record["label"], "Probability": record["probabilities"][label_index]})
                st.dataframe(pd.DataFrame(output_rows), width="stretch", hide_index=True)
                st.caption("This final-model row inference is illustrative; it is not a held-out prediction or key-safety assessment.")
            else:
                st.info("Select a sample window and run classification to show its model outputs.")

    with st.container(border=True):
        st.subheader("Model Performance Metrics")
        track_records = performance_frame(summary.get("summary", []), model_track)
        if not track_records.empty:
            st.dataframe(
                track_records.style.format({
                    "Accuracy": "{:.3f}",
                    "Macro-F1": "{:.3f}",
                    "Balanced accuracy": "{:.3f}",
                    "Log loss": "{:.3f}",
                    "MGV precision": "{:.2e}",
                    "MGV recall": "{:.3f}",
                    "MGV mean fold support": "{:.0f}",
                }),
                width="stretch",
                hide_index=True,
            )
            plot = track_records[["Model", "Macro-F1", "Accuracy", "Balanced accuracy"]].melt(
                id_vars="Model", var_name="Metric", value_name="Score",
            )
            chart = alt.Chart(plot).mark_bar().encode(
                x=alt.X("Model:N", title=None),
                y=alt.Y("Score:Q", scale=alt.Scale(domain=[0, 1]), title="Weighted fold-mean score"),
                color=alt.Color("Metric:N", scale=alt.Scale(scheme="blues")),
                xOffset="Metric:N",
                tooltip=["Model", "Metric", alt.Tooltip("Score:Q", format=".3f")],
            ).properties(height=260)
            st.altair_chart(chart, width="stretch")
            st.caption("Five-fold spatially grouped metrics; weighted sample, two-window purge. The uncalibrated soft-voting ensemble's macro-F1 is about 0.544.")
        else:
            st.info("Saved model comparison metrics were not found.")


def run_local_shap(model, x: pd.DataFrame, class_index: int) -> np.ndarray:
    import shap
    explainer = shap.TreeExplainer(model.get_booster())
    values = explainer.shap_values(x)
    if isinstance(values, list):
        return np.asarray(values[class_index])[0].astype(float)
    array = np.asarray(values)
    if array.ndim == 3:
        return array[0, :, class_index].astype(float)
    if array.ndim == 2:
        return array[0].astype(float)
    raise ValueError(f"unrecognized SHAP output shape: {array.shape}")


def draw_shap_tab(eval_df: pd.DataFrame | None, shap_summary: dict, demo: dict) -> None:
    left, right = st.columns([0.9, 1.8], gap="large")
    selected_id = int(st.session_state.get("selected_window_id", demo.get("window_id", 0)))
    row = selected_evaluation_row(eval_df, selected_id) if eval_df is not None else None
    local_values = None
    selected_class = "CC"
    if row is not None:
        x = pd.DataFrame([row[list(MODEL_FEATURE_NAMES)].astype(float).to_dict()], columns=MODEL_FEATURE_NAMES)
    else:
        x = None

    with left:
        with st.container(border=True):
            st.subheader("SHAP Analysis")
            st.metric("Selected Window", f"{selected_id:,}")
            if row is not None:
                st.write("**Observed target:**", row.get("label_adviser_classical_4class", "—"))
            st.caption("The SHAP explainer uses the saved final XGBoost model. Explanations describe model behavior, not causes.")
            run_shap = st.button("Run SHAP Analysis", type="primary", width="stretch", disabled=x is None)
            if run_shap and x is not None:
                try:
                    models, errors = load_final_models(str(PROJECT_ROOT))
                    xgb_model = models.get("XGBoost")
                    if xgb_model is None:
                        raise RuntimeError(errors.get("XGBoost", "XGBoost artifact unavailable"))
                    predicted, _ = predict_one(xgb_model, x)
                    selected_class = predicted
                    local_values = run_local_shap(xgb_model, x, CLASS_NAMES.index(predicted))
                    st.session_state["local_shap_result"] = {
                        "window_id": selected_id,
                        "class": predicted,
                        "values": local_values.tolist(),
                    }
                except Exception as exc:
                    st.warning(f"Local SHAP is unavailable in this environment: {type(exc).__name__}: {exc}")
            saved_shap = st.session_state.get("local_shap_result")
            if saved_shap and saved_shap["window_id"] == selected_id:
                selected_class = saved_shap["class"]
                local_values = np.asarray(saved_shap["values"], dtype=float)
                st.write("**Predicted Regime:**", selected_class)
                st.caption("Local SHAP values shown for the predicted class.")
            with st.container(border=True):
                st.markdown("**Interpretation**")
                st.write("Features with the largest absolute SHAP values contributed most to this model output. This does not show causal influence, security strength, or a reliable MGV detector.")

    with right:
        with st.container(border=True):
            st.subheader("FEATURE IMPORTANCE")
            names = shap_summary.get("features", [])
            values = shap_summary.get("mean_absolute_shap", [])
            global_df = pd.DataFrame({"Feature": names, "Global mean |SHAP|": values})
            if local_values is not None and len(local_values) == len(MODEL_FEATURE_NAMES):
                global_df = global_df[global_df["Feature"].isin(MODEL_FEATURE_NAMES)].copy()
                local_map = dict(zip(MODEL_FEATURE_NAMES, local_values))
                global_df["Selected window value"] = [float(x.iloc[0][name]) for name in global_df["Feature"]]
                global_df["Local SHAP impact"] = [local_map[name] for name in global_df["Feature"]]
                global_df = global_df.reindex(global_df["Local SHAP impact"].abs().sort_values(ascending=False).index)
                plot_df = global_df.head(12).sort_values("Local SHAP impact")
                chart = alt.Chart(plot_df).mark_bar().encode(
                    x=alt.X("Local SHAP impact:Q", title=f"Local SHAP impact · {selected_class}"),
                    y=alt.Y("Feature:N", sort="-x", title=None),
                    color=alt.condition("datum['Local SHAP impact'] >= 0", alt.value("#354253"), alt.value("#8a98a8")),
                    tooltip=["Feature", "Selected window value", "Local SHAP impact", "Global mean |SHAP|"],
                ).properties(height=360)
                st.altair_chart(chart, width="stretch")
                st.dataframe(global_df.head(20), width="stretch", hide_index=True)
            elif not global_df.empty:
                global_df = global_df.sort_values("Global mean |SHAP|", ascending=False).head(12)
                st.altair_chart(
                    alt.Chart(global_df).mark_bar(color="#354253").encode(
                        x=alt.X("Global mean |SHAP|:Q", title="Mean absolute SHAP value"),
                        y=alt.Y("Feature:N", sort="-x", title=None),
                        tooltip=["Feature", "Global mean |SHAP|"],
                    ).properties(height=360),
                    width="stretch",
                )
                st.dataframe(global_df, width="stretch", hide_index=True)
                st.caption(f"Saved global SHAP summary over {shap_summary.get('data_rows_explained', 0):,} sampled rows. Run SHAP Analysis for the selected window's local values.")
            else:
                st.info("No SHAP summary is bundled with this project copy.")


def _redacted_keygen_record(result, status: str, error: str | None = None) -> dict:
    """Keep only panel-safe key-generation output; never persist the key object."""
    candidates = []
    if result is not None:
        for candidate in result.candidates:
            eligibility = candidate.eligibility
            if eligibility is None:
                continue
            candidates.append({
                "Attempt": candidate.attempt,
                "Decision": "accepted" if candidate.accepted else "rejected",
                "P(EVENT)": round(eligibility.p_event, 4),
                "P(CC)": round(eligibility.p_cc, 4),
                "ML class": eligibility.final_class,
                "CR": round(eligibility.cr, 4),
                "KS pass": eligibility.ks_pass,
                "Outside training range": eligibility.outside_training_range,
            })
        summary = result.redacted_info()
    else:
        summary = {}
    return {"status": status, "error": error, "summary": summary, "candidates": candidates}


def draw_pgi_tab(crypto: dict, manifest: dict, pgi_progress: dict) -> None:
    left, right = st.columns([0.95, 1.8], gap="large")
    with left:
        with st.container(border=True):
            st.subheader("Experimental RSA-2048 Run")
            if st.button("Run Experimental RSA-2048 Demo", type="primary", width="stretch"):
                try:
                    with st.spinner("Generating and validating an experimental RSA-2048 result. This may take about a minute…"):
                        result = PGICryptoKeyGenerator().generate(
                            2048,
                            policy=POLICY_EXPERIMENTAL,
                            pool_size=2,
                            max_attempts=8,
                            rng_mode=RNG_MODE_SECURE,
                            validate=True,
                        )
                    st.session_state["pgi_experimental_run"] = _redacted_keygen_record(result, "success")
                    del result
                except KeyGenerationError as exc:
                    partial = getattr(exc, "result", None)
                    st.session_state["pgi_experimental_run"] = _redacted_keygen_record(
                        partial, "failed", str(exc)
                    )
                except Exception as exc:
                    st.session_state["pgi_experimental_run"] = _redacted_keygen_record(
                        None, "error", f"{type(exc).__name__}: {exc}"
                    )

            run = st.session_state.get("pgi_experimental_run")
            if run:
                if run["status"] == "success":
                    st.success("Experimental run completed and the library validation passed. ML eligibility was not enforced.")
                elif run["status"] == "failed":
                    st.error("Experimental run did not produce a validated key pair.")
                    if run.get("error"):
                        st.caption(run["error"])
                else:
                    st.error("Experimental run could not start or complete.")
                    if run.get("error"):
                        st.caption(run["error"])
                summary = run.get("summary", {})
                if summary:
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Attempts", summary.get("candidate_attempts", "—"))
                    c2.metric("Generation time", f"{summary.get('generation_time_seconds', 0):.2f} s")
                    validation = summary.get("validation") or "Not available"
                    c3.metric("Validation", validation)
                    st.caption("Key: REDACTED · p, q, d, CRT values: REDACTED · private material is not saved or exported.")
                    if run.get("candidates"):
                        st.dataframe(pd.DataFrame(run["candidates"]), width="stretch", hide_index=True)
                    for warning in summary.get("warnings", []):
                        st.caption(warning)
                elif run.get("error"):
                    st.caption(run["error"])
    with right:
        with st.container(border=True):
            st.subheader("Standard Cryptographic Baselines")
            results = crypto.get("results", [])
            rows = []
            for item in results:
                if item.get("algorithm") in {"RSA-2048", "RSA-4096", "DH-2048", "ECC-P-256"}:
                    rows.append({
                        "Algorithm": item.get("algorithm"),
                        "Trials": item.get("trials"),
                        "Mean time (s)": item.get("mean_seconds"),
                        "Median time (s)": item.get("median_seconds"),
                        "95th percentile (s)": item.get("p95_seconds"),
                    })
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
            st.caption("Standard-library timings are measured separately from PGI Experimental trials; compare them as prototype pipeline timings, not as a security advantage.")
        with st.container(border=True):
            st.subheader("Model Ranges")
            st.metric("Observed model support", f"{manifest.get('observed_p_mid_min', 233):,} – {manifest.get('observed_p_mid_max', 0):,}")
            st.write("RSA-2048 factor primes fall far outside this observed support. Experimental predictions are descriptive only; they do not validate key selection or establish improved security.")
        with st.container(border=True):
            st.subheader("Experimental 1,000-Trial Status")
            if pgi_progress:
                c1, c2, c3 = st.columns(3)
                c1.metric("Trials completed", f"{pgi_progress.get('completed_trials', 0):,} / {pgi_progress.get('requested_trials', 1000):,}")
                c2.metric("Validated key pairs", f"{pgi_progress.get('successful_key_pairs', 0):,}")
                c3.metric("Mean time", f"{pgi_progress.get('mean_seconds_completed', 0):.2f} s")
                st.caption("Fresh-model predictions are out of range and are reported, not enforced. No private key material is saved.")
            else:
                st.info("The long Experimental series has not reached its first ten-trial checkpoint yet.")


def main() -> None:
    show_main_header()
    tabs = st.tabs(["Data", "Classification", "SHAP", "PGI-Crypto"])
    dataset_meta = json_record("dataset_summary.json", "data/raw/prime_gaps_1e12.u16.meta.json")
    model_summary = json_record("model_comparison_summary.json", "results/metrics/full_1e12/model_comparison_summary.json")
    shap_summary = json_record("shap_global.json", "results/metrics/full_1e12/shap_adviser_classical_4class_global.json")
    crypto_summary = json_record("crypto_comparison.json", "results/benchmarks/full_1e12_crypto_comparison.json")
    pgi_progress = json_record("pgi_experimental_progress.json", "results/benchmarks/pgi_experimental_1000/summary.partial.json")
    pgi_progress = normalize_experimental_progress(pgi_progress)
    model_manifest = json_record("model_manifest.json")
    demo = json_record("demo_window.json")
    feature_df = parquet_frame(FEATURE_SAMPLE_PATH)
    evaluation_df = parquet_frame(MODEL_SAMPLE_PATH)

    with tabs[0]:
        draw_data_tab(feature_df, dataset_meta)
    with tabs[1]:
        draw_classification_tab(evaluation_df, model_summary, demo)
    with tabs[2]:
        draw_shap_tab(evaluation_df, shap_summary, demo)
    with tabs[3]:
        draw_pgi_tab(crypto_summary, model_manifest, pgi_progress)

    st.markdown("<hr style='margin-top:2rem;border-color:#263349'>", unsafe_allow_html=True)
    st.markdown(
        "<div class='app-subtitle'>PGI-Crypto research prototype · predictions are not key-safety decisions</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()

