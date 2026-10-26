(() => {
  const $ = (selector) => document.querySelector(selector);
  const number = (value) => Number(value || 0).toLocaleString("en-US");
  const percent = (value, digits = 1) => value == null ? "—" : `${(Number(value) * 100).toFixed(digits)}%`;
  const score = (value) => value == null ? "—" : Number(value).toFixed(3);
  const seconds = (value) => {
    if (value == null) return "—";
    const n = Number(value);
    if (n < 0.001) return `${(n * 1e6).toFixed(1)} µs`;
    if (n < 1) return `${(n * 1000).toFixed(2)} ms`;
    return `${n.toFixed(3)} s`;
  };
  const escapeHtml = (value) => String(value).replace(/[&<>"']/g, (char) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;"
  })[char]);

  function showPage(name) {
    document.querySelectorAll(".page").forEach((page) => page.classList.toggle("active", page.id === `page-${name}`));
    document.querySelectorAll(".nav-link").forEach((link) => link.classList.toggle("active", link.dataset.page === name));
    const active = document.querySelector(`.nav-link[data-page="${name}"]`);
    $("#current-section").textContent = active ? active.textContent.trim().toUpperCase() : name.toUpperCase();
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  document.querySelectorAll("[data-page]").forEach((control) => {
    control.addEventListener("click", () => showPage(control.dataset.page));
  });

  async function getJson(path) {
    const response = await fetch(path, { headers: { Accept: "application/json" } });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || `Request failed (${response.status})`);
    return data;
  }

  function modelRowsForTrack(rows, trackPart) {
    return rows.filter((row) => row.method.includes(trackPart));
  }

  function renderModelBars(summary) {
    const rows = modelRowsForTrack(summary.evaluation.model_rows, "label_adviser_classical_4class")
      .sort((a, b) => b.macro_f1 - a.macro_f1);
    const best = rows[0];
    $("#overview-model-bars").innerHTML = rows.map((row) => `
      <div class="bar-row">
        <span class="bar-label" title="${escapeHtml(row.model)}">${escapeHtml(row.model.replace("_uncalibrated", ""))}</span>
        <div class="bar-track"><div class="bar-fill" style="width:${Math.max(0, Math.min(100, Number(row.macro_f1) * 100)).toFixed(1)}%"></div></div>
        <span class="bar-value">${score(row.macro_f1)}</span>
      </div>`).join("");
    $("#metric-f1").textContent = best ? score(best.macro_f1) : "—";
    const ensemble = rows.find((row) => row.model.includes("SoftVoting")) || best;
    if (ensemble) {
      $("#mgv-precision").textContent = percent(ensemble.mgv_precision, 3);
      $("#mgv-recall").textContent = percent(ensemble.mgv_recall, 1);
      $("#mgv-precision-detail").textContent = percent(ensemble.mgv_precision, 3);
      $("#mgv-recall-detail").textContent = percent(ensemble.mgv_recall, 1);
      $("#mgv-support-detail").textContent = number(ensemble.mgv_support);
    }
  }

  function renderModelTable(rows) {
    const order = { SoftVotingEnsemble_uncalibrated: 0, XGBoost: 1, RandomForest: 2, MLP: 3 };
    const sorted = [...rows].sort((a, b) => {
      const track = a.method.localeCompare(b.method);
      return track || (order[a.model] ?? 10) - (order[b.model] ?? 10);
    });
    $("#model-table-body").innerHTML = sorted.map((row) => `
      <tr>
        <td>${escapeHtml(row.method.replace("label_", "").replaceAll("_", " "))}</td>
        <td>${escapeHtml(row.model.replace("_uncalibrated", " (uncalibrated)"))}</td>
        <td>${score(row.accuracy)}</td><td><strong>${score(row.macro_f1)}</strong></td>
        <td>${score(row.balanced_accuracy)}</td><td>${percent(row.mgv_precision, 3)}</td>
        <td>${percent(row.mgv_recall, 1)}</td>
      </tr>`).join("");
  }

  function renderShap(items) {
    const maxValue = Math.max(...items.map((item) => Number(item.mean_absolute_shap)), 1);
    $("#shap-bars").innerHTML = items.map((item) => `
      <div class="shap-row"><span class="shap-name" title="${escapeHtml(item.feature)}">${escapeHtml(item.feature)}</span>
        <div class="shap-track"><div class="shap-fill" style="width:${(Number(item.mean_absolute_shap) / maxValue * 100).toFixed(1)}%"></div></div>
        <span class="shap-value">${Number(item.mean_absolute_shap).toFixed(3)}</span>
      </div>`).join("");
  }

  function renderCrypto(summary) {
    $("#trial-note").textContent = `${number(summary.crypto.trial_count_per_baseline)} trials per baseline`;
    $("#crypto-table-body").innerHTML = summary.crypto.baselines.map((row) => `
      <tr><td>${escapeHtml(row.algorithm)}</td><td>${number(row.key_size_bits)} bits</td><td>${number(row.trials)}</td>
      <td>${seconds(row.mean_seconds)}</td><td>${seconds(row.median_seconds)}</td><td>${seconds(row.p95_seconds)}</td></tr>`).join("");
    $("#strict-status").textContent = String(summary.crypto.strict_pgi_status || "not recorded").replaceAll("_", " ");
    $("#strict-reason").textContent = `${summary.crypto.strict_pgi_reason} No PGI key pair was produced by the strict range check.`;
  }

  function renderSummary(summary) {
    $("#system-status").textContent = "Study records loaded · local only";
    $("#metric-gaps").textContent = number(summary.dataset.gap_count);
    $("#metric-windows").textContent = number(summary.dataset.window_population);
    $("#metric-rows").textContent = number(summary.dataset.model_evaluation_rows);
    $("#evaluation-size-pill").textContent = `${number(summary.dataset.model_evaluation_rows)} rows`;
    $("#gpu-name").textContent = summary.hardware.gpu_name || "GPU not recorded";
    $("#gpu-state").textContent = summary.hardware.xgboost_cuda_fit ? `CUDA verified · XGBoost ${summary.hardware.xgboost_version}` : "CUDA training not recorded";
    $("#gpu-fit").textContent = summary.hardware.xgboost_cuda_fit ? "Confirmed" : "Not confirmed";
    $("#gpu-check").textContent = summary.hardware.xgboost_cuda_fit ? "✓" : "—";
    $("#model-file-label").textContent = summary.project.model_file;
    $("#model-hash").textContent = summary.project.model_sha256;
    $("#method-data-detail").textContent = `${number(summary.dataset.prime_count)} primes · ${number(summary.dataset.gap_count)} gaps · ${number(summary.dataset.bytes)} raw bytes · max gap ${summary.dataset.max_gap}`;
    $("#method-window-detail").textContent = `${number(summary.dataset.feature_sample_rows)} feature rows · ${number(summary.dataset.model_evaluation_rows)} evaluation rows · population weight ${number(summary.dataset.window_population)}`;
    $("#method-model-detail").textContent = `${summary.project.model_features} model inputs · ${summary.evaluation.cross_validation} · device: ${summary.evaluation.device_used}`;
    renderModelBars(summary);
    renderModelTable(summary.evaluation.model_rows);
    renderCrypto(summary);
    renderShap(summary.explainability.top_features);
  }

  function setGapCount() {
    const tokens = $("#gap-values").value.trim().split(/[\s,;]+/).filter(Boolean);
    $("#gap-count").textContent = `${tokens.length} / 100`;
    $("#gap-count").style.color = tokens.length === 100 ? "#17866e" : "";
  }

  async function loadExample() {
    try {
      const sample = await getJson("/api/example");
      $("#p-mid").value = sample.p_mid;
      $("#gap-values").value = sample.gaps.join(", ");
      $("#example-source").textContent = sample.source;
      $("#lab-error").textContent = "";
      setGapCount();
    } catch (error) {
      $("#example-source").textContent = "The built-in example could not be loaded.";
      $("#lab-error").textContent = error.message;
    }
  }

  function renderPrediction(result) {
    const colors = { CC: "#4c7de8", MA: "#9a6bd0", MGV: "#cf8553", NONE: "#7d91a7" };
    const probabilities = Object.entries(result.probabilities).sort((a, b) => b[1] - a[1]);
    const features = result.selected_features;
    $("#empty-result").classList.add("hidden");
    const output = $("#classification-result");
    output.classList.remove("hidden");
    output.innerHTML = `
      <div class="result-label-box"><div><span>Predicted regime</span><br><strong>${escapeHtml(result.predicted_label)}</strong></div><span class="confidence-chip">max probability ${percent(result.confidence, 1)}</span></div>
      <div class="probability-list">${probabilities.map(([label, value]) => `
        <div class="probability-row"><span>${escapeHtml(label)}</span><div class="prob-track"><div class="prob-fill" style="width:${(value * 100).toFixed(2)}%;background:${colors[label] || "#6d90e8"}"></div></div><strong>${percent(value, 2)}</strong></div>`).join("")}</div>
      <div class="result-stats">
        <div class="result-stat"><span>Cramér ratio (descriptive)</span><strong>${Number(result.descriptive_statistics.cr).toFixed(4)}</strong></div>
        <div class="result-stat"><span>KS statistic</span><strong>${Number(result.descriptive_statistics.ks_statistic).toFixed(4)}</strong></div>
        <div class="result-stat"><span>KS p-value</span><strong>${Number(result.descriptive_statistics.ks_pvalue).toExponential(2)}</strong></div>
        <div class="result-stat"><span>Mean gap</span><strong>${Number(features.mean_gap).toFixed(2)}</strong></div>
      </div>
      <div class="window-range">Window span: <strong>${number(result.window.p_start)}</strong> → <strong>${number(result.window.p_end)}</strong> · centered on <strong>${number(result.window.p_mid)}</strong></div>
      <p class="small-note">The maximum class probability is not described as calibrated confidence. ${escapeHtml(result.interpretation)}</p>`;
  }

  async function classifyWindow() {
    const button = $("#classify-button");
    const errorBox = $("#lab-error");
    errorBox.textContent = "";
    const midpoint = Number($("#p-mid").value);
    const tokens = $("#gap-values").value.trim().split(/[\s,;]+/).filter(Boolean);
    const gaps = tokens.map((token) => Number(token));
    if (!Number.isSafeInteger(midpoint)) {
      errorBox.textContent = "Enter a whole-number midpoint prime.";
      return;
    }
    if (gaps.length !== 100 || gaps.some((gap) => !Number.isSafeInteger(gap))) {
      errorBox.textContent = `Enter exactly 100 integer gap values. Current count: ${gaps.length}.`;
      return;
    }
    button.disabled = true;
    button.querySelector("span:first-child").textContent = "Analyzing window…";
    try {
      const response = await fetch("/api/classify", {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "application/json" },
        body: JSON.stringify({ p_mid: midpoint, gaps })
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || `Analysis failed (${response.status})`);
      renderPrediction(result);
    } catch (error) {
      errorBox.textContent = error.message;
    } finally {
      button.disabled = false;
      button.querySelector("span:first-child").textContent = "Classify this window";
    }
  }

  $("#gap-values").addEventListener("input", setGapCount);
  $("#load-example").addEventListener("click", loadExample);
  $("#classify-button").addEventListener("click", classifyWindow);

  Promise.all([getJson("/api/summary"), loadExample()])
    .then(([summary]) => renderSummary(summary))
    .catch((error) => {
      $("#system-status").textContent = "Study records unavailable";
      $("#overview-model-bars").innerHTML = `<div class="form-error">${escapeHtml(error.message)}</div>`;
    });
})();
