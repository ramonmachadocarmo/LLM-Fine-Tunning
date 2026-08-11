const state = {
  assets: null,
  selectedDatasets: new Set(),
  currentJobId: null,
  lastConfigPath: null,
  pollTimer: null,
  tab: "setup",
  chatMessages: [],
  explorer: {
    target: null,
    path: ".",
    selected: null,
    mode: "all",
    extensions: null,
  },
};

const t = (key, vars) => (window.FT_I18N ? window.FT_I18N.t(key, vars) : key);

function browseTargets() {
  return {
    config: {
      title: t("browse.config"),
      mode: "files",
      extensions: ".yaml,.yml",
      start: "configs",
      onSelect: async (path) => {
        $("#load-config").value = path;
        await loadSelectedConfig();
      },
    },
    output_dir: {
      title: t("browse.output_dir"),
      mode: "dirs",
      extensions: null,
      start: "adapters",
      onSelect: (path) => {
        $("#output_dir").value = `./${path}`;
        if (!$("#adapter_path").value) setExportPath("adapter_path", `./${path}`);
      },
    },
    base_model: {
      title: t("browse.base_model"),
      mode: "dirs",
      extensions: null,
      start: "merged_models",
      onSelect: (path) => {
        $("#base_model").value = `./${path}`;
        $("#local-model").value = path;
      },
    },
    datasets: {
      title: t("browse.datasets"),
      mode: "files",
      extensions: ".jsonl",
      start: "data",
      onSelect: (path) => {
        state.selectedDatasets.add(path);
        renderDatasets(state.assets?.datasets || []);
        renderSelectedDatasets();
        updateSetupSummary();
      },
    },
    adapter_path: {
      title: t("browse.adapter_path"),
      mode: "dirs",
      extensions: null,
      start: "adapters",
      onSelect: (path) => setExportPath("adapter_path", `./${path}`),
    },
    merged_path: {
      title: t("browse.merged_path"),
      mode: "dirs",
      extensions: null,
      start: "models",
      onSelect: (path) => setExportPath("merged_path", `./${path}`),
    },
    gguf_filename: {
      title: t("browse.gguf_filename"),
      mode: "dirs",
      extensions: null,
      start: "models",
      onSelect: (path) => {
        const name = ($("#project_name").value || "experiment").trim() || "experiment";
        setExportPath("gguf_filename", `./${path}/${name}.gguf`);
      },
    },
  };
}

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

function setExportPath(key, value) {
  const hidden = $(`#${key}`);
  const ui = $(`#${key}_ui`);
  if (hidden) hidden.value = value;
  if (ui) ui.value = value;
}

function syncExportFieldsFromUi() {
  for (const key of ["adapter_path", "merged_path", "gguf_filename"]) {
    const ui = $(`#${key}_ui`);
    const hidden = $(`#${key}`);
    if (ui && hidden && ui.value.trim()) hidden.value = ui.value.trim();
  }
}

function syncExportUiFromHidden() {
  for (const key of ["adapter_path", "merged_path", "gguf_filename"]) {
    const ui = $(`#${key}_ui`);
    const hidden = $(`#${key}`);
    if (ui && hidden) ui.value = hidden.value || "";
  }
}

function setStatus(msg, kind = "") {
  const el = $("#form-status");
  if (el) {
    el.textContent = msg || "";
    el.className = `status ${kind}`.trim();
  }
  const train = $("#train-status");
  if (train && state.tab === "train") {
    train.textContent = msg || "";
    train.className = `status ${kind}`.trim();
  }
  const exp = $("#export-status");
  if (exp && state.tab === "export") {
    exp.textContent = msg || "";
    exp.className = `status ${kind}`.trim();
  }
}

function updateSetupSummary() {
  const dash = t("msg.dash");
  const set = (id, val) => {
    const el = $(id);
    if (el) el.textContent = val || dash;
  };
  set("#sum-project", $("#project_name")?.value);
  set("#sum-model", $("#base_model")?.value);
  set("#sum-datasets", String(state.selectedDatasets.size));
  set("#sum-config", state.lastConfigPath || $("#load-config")?.value || dash);
  const runProject = $("#run-project");
  const runMeta = $("#run-meta");
  if (runProject) runProject.textContent = $("#project_name")?.value || dash;
  if (runMeta) {
    const ds = state.selectedDatasets.size;
    const model = $("#base_model")?.value || dash;
    runMeta.textContent = ds
      ? t("msg.run_meta", { n: ds, model })
      : t("train.configure_first");
  }
}

function syncDerivedPaths() {
  const name =
    ($("#project_name").value || "experiment")
      .trim()
      .replace(/[^a-zA-Z0-9_\-]+/g, "_")
      .toLowerCase() || "experiment";
  if (!$("#output_dir").value) $("#output_dir").value = `./adapters/${name}`;
  if (!$("#adapter_path").value) setExportPath("adapter_path", $("#output_dir").value || `./adapters/${name}`);
  if (!$("#merged_path").value) setExportPath("merged_path", `./models/${name}_merged`);
  if (!$("#gguf_filename").value) setExportPath("gguf_filename", `./models/${name}.gguf`);
  syncExportUiFromHidden();
  updateSetupSummary();
}

function formPayload() {
  syncExportFieldsFromUi();
  syncDerivedPaths();
  const maxStepsRaw = $("#max_steps").value;
  return {
    project_name: $("#project_name").value.trim(),
    base_model: $("#base_model").value.trim(),
    dataset_paths: Array.from(state.selectedDatasets),
    load_in_4bit: $("#load_in_4bit").checked,
    epochs: Number($("#epochs").value),
    batch_size: Number($("#batch_size").value),
    gradient_accumulation_steps: Number($("#gradient_accumulation_steps").value),
    save_strategy: $("#save_strategy").value,
    save_steps: Number($("#save_steps").value),
    learning_rate: Number($("#learning_rate").value),
    max_seq_length: Number($("#max_seq_length").value),
    max_steps: maxStepsRaw === "" ? null : Number(maxStepsRaw),
    system_prompt: $("#system_prompt").value,
    output_dir: $("#output_dir").value.trim() || null,
    adapter_path: $("#adapter_path").value.trim() || null,
    merged_path: $("#merged_path").value.trim() || null,
    gguf_filename: $("#gguf_filename").value.trim() || null,
    save_config_as: $("#project_name").value.trim(),
  };
}

function fillConfig(cfg) {
  $("#project_name").value = cfg.project?.name || "";
  $("#output_dir").value = cfg.project?.output_dir || "";
  $("#base_model").value = cfg.model?.base_model || "";
  $("#load_in_4bit").checked = Boolean(cfg.model?.load_in_4bit ?? true);

  const training = cfg.training || {};
  $("#epochs").value = training.epochs ?? 2;
  $("#batch_size").value = training.batch_size ?? 2;
  $("#gradient_accumulation_steps").value = training.gradient_accumulation_steps ?? 8;
  $("#learning_rate").value = training.learning_rate ?? 0.0002;
  $("#max_seq_length").value = training.max_seq_length ?? 1024;
  $("#save_strategy").value = training.save_strategy || "steps";
  $("#save_steps").value = training.save_steps ?? 50;
  $("#max_steps").value = training.max_steps ?? "";
  $("#system_prompt").value = cfg.system_prompt || "";

  setExportPath("adapter_path", cfg.export?.adapter_path || "");
  setExportPath("merged_path", cfg.export?.merged_path || "");
  setExportPath("gguf_filename", cfg.export?.gguf_filename || "");

  state.selectedDatasets = new Set(training.dataset_paths || []);
  renderDatasets(state.assets?.datasets || []);
  renderSelectedDatasets();
  updateSetupSummary();
}

function renderSelectedDatasets() {
  const root = $("#selected-datasets");
  root.innerHTML = "";
  if (!state.selectedDatasets.size) {
    root.innerHTML = `<span class="hint">${t("msg.no_dataset_selected")}</span>`;
    return;
  }
  for (const path of state.selectedDatasets) {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "path-chip";
    chip.innerHTML = `<code>${path}</code><span aria-hidden="true">×</span>`;
    chip.title = t("msg.remove");
    chip.addEventListener("click", () => {
      state.selectedDatasets.delete(path);
      renderDatasets(state.assets?.datasets || []);
      renderSelectedDatasets();
      updateSetupSummary();
    });
    root.appendChild(chip);
  }
}

function renderDatasets(datasets) {
  const root = $("#dataset-list");
  root.innerHTML = "";
  if (!datasets.length) {
    root.innerHTML = `<p class="hint">${t("msg.no_jsonl")}</p>`;
    return;
  }
  for (const ds of datasets) {
    const label = document.createElement("label");
    label.className = `chip ${state.selectedDatasets.has(ds.path) ? "active" : ""}`;
    const tag = ds.kind === "sample" ? " · sample" : "";
    label.innerHTML = `<input type="checkbox" value="${ds.path}" ${
      state.selectedDatasets.has(ds.path) ? "checked" : ""
    } /><span>${ds.name}${tag}</span>`;
    label.querySelector("input").addEventListener("change", (e) => {
      if (e.target.checked) state.selectedDatasets.add(ds.path);
      else state.selectedDatasets.delete(ds.path);
      label.classList.toggle("active", e.target.checked);
      renderSelectedDatasets();
      updateSetupSummary();
    });
    root.appendChild(label);
  }
}

function renderAssets(assets) {
  state.assets = assets;

  const cfgSelect = $("#load-config");
  const current = cfgSelect.value;
  cfgSelect.innerHTML = `<option value="">${t("msg.dash")}</option>`;
  for (const c of assets.configs || []) {
    const opt = document.createElement("option");
    opt.value = c.path;
    const tag = c.kind === "template" ? " [template]" : "";
    opt.textContent = `${c.path}${tag}`;
    cfgSelect.appendChild(opt);
  }
  cfgSelect.value = current;

  const local = $("#local-model");
  local.innerHTML = `<option value="">${t("setup.local_model_manual")}</option>`;
  for (const m of assets.local_models || []) {
    if (m.kind !== "local_hf") continue;
    const opt = document.createElement("option");
    opt.value = m.path;
    opt.textContent = `${m.name} (${m.path})`;
    local.appendChild(opt);
  }

  const dl = $("#hf-suggestions");
  dl.innerHTML = "";
  for (const id of assets.suggested_hf_models || []) {
    const opt = document.createElement("option");
    opt.value = id;
    dl.appendChild(opt);
  }

  const adapters = $("#adapter-list");
  adapters.innerHTML = "";
  if (!(assets.adapters || []).length) {
    adapters.innerHTML = `<li>${t("msg.no_adapters")}</li>`;
  } else {
    for (const a of assets.adapters) {
      const li = document.createElement("li");
      li.innerHTML = `<strong>${a.name}</strong><br/><code>${a.path}</code>${a.has_best_model ? " · best" : ""}`;
      adapters.appendChild(li);
    }
  }

  renderDatasets(assets.datasets || []);
  renderSelectedDatasets();
}

async function refreshHfTokenStatus() {
  const el = $("#hf-token-status");
  if (!el) return;
  try {
    const st = await api("/api/hf/token");
    if (st.configured) {
      el.textContent = t("setup.hf_token_status_ok", {
        masked: st.masked || "••••",
        source: st.source,
      });
      el.className = "hint status ok";
    } else {
      el.textContent = t("setup.hf_token_status_none");
      el.className = "hint";
    }
  } catch {
    el.textContent = t("setup.hf_token_hint");
    el.className = "hint";
  }
}

async function saveHfToken() {
  const token = $("#hf_token")?.value?.trim() || "";
  if (!token) {
    setStatus(t("msg.hf_token_need"), "bad");
    return;
  }
  await api("/api/hf/token", {
    method: "PUT",
    body: JSON.stringify({ token }),
  });
  if ($("#hf_token")) $("#hf_token").value = "";
  await refreshHfTokenStatus();
  setStatus(t("msg.hf_token_saved"), "ok");
}

async function clearHfToken() {
  await api("/api/hf/token", { method: "DELETE" });
  if ($("#hf_token")) $("#hf_token").value = "";
  await refreshHfTokenStatus();
  setStatus(t("msg.hf_token_cleared"), "ok");
}

async function validateBaseModel() {
  const ref = $("#base_model")?.value?.trim() || "";
  const statusEl = $("#model-check-status");
  if (!ref) {
    setStatus(t("msg.model_need_ref"), "bad");
    if (statusEl) {
      statusEl.hidden = false;
      statusEl.className = "hint status bad";
      statusEl.textContent = t("msg.model_need_ref");
    }
    return;
  }
  setStatus(t("msg.validating_model"));
  if (statusEl) {
    statusEl.hidden = false;
    statusEl.className = "hint";
    statusEl.textContent = t("msg.validating_model");
  }
  try {
    const data = await api("/api/models/validate", {
      method: "POST",
      body: JSON.stringify({ base_model: ref }),
    });
    const msg = t("msg.model_ok", { kind: data.kind, checksum: data.checksum });
    const detail = data.detail ? `${msg} — ${data.detail}` : msg;
    setStatus(detail, "ok");
    if (statusEl) {
      statusEl.className = "hint status ok";
      statusEl.textContent = detail;
    }
  } catch (err) {
    setStatus(err.message, "bad");
    if (statusEl) {
      statusEl.className = "hint status bad";
      statusEl.textContent = err.message;
    }
  }
}

async function api(path, options) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options?.headers || {}) },
    ...options,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const detail = Array.isArray(data.detail)
      ? data.detail.map((d) => d.msg || JSON.stringify(d)).join("; ")
      : data.detail || res.statusText || t("msg.request_failed");
    throw new Error(detail);
  }
  return data;
}

async function refreshAssets() {
  const assets = await api("/api/assets");
  renderAssets(assets);
}

async function loadSelectedConfig() {
  const path = $("#load-config").value;
  if (!path) return;
  const data = await api(`/api/configs/${encodeURI(path)}`);
  fillConfig(data.config);
  state.lastConfigPath = path;
  $("#btn-export").disabled = false;
  setStatus(t("msg.config_loaded", { path }), "ok");
}

async function openExplorer(targetKey) {
  const conf = browseTargets()[targetKey];
  if (!conf) return;
  state.explorer = {
    target: targetKey,
    path: conf.start || ".",
    selected: null,
    mode: conf.mode,
    extensions: conf.extensions,
  };
  $("#explorer-title").textContent = conf.title;
  $("#explorer-select").disabled = true;
  $("#explorer").showModal();
  await loadExplorer();
}

async function loadExplorer() {
  const ex = state.explorer;
  const params = new URLSearchParams({
    path: ex.path === "." ? "" : ex.path,
    mode: ex.mode,
  });
  if (ex.extensions) params.set("extensions", ex.extensions);
  const data = await api(`/api/browse?${params}`);
  $("#explorer-current").textContent = data.current;
  $("#explorer-up").disabled = data.current === ".";
  state.explorer.parent = data.parent;
  state.explorer.selected = null;
  $("#explorer-select").disabled = true;

  const list = $("#explorer-list");
  list.innerHTML = "";

  for (const entry of data.entries) {
    const li = document.createElement("li");
    li.className = `explorer-item ${entry.is_dir ? "dir" : "file"}`;
    li.innerHTML = `<span class="ico">${entry.is_dir ? "📁" : "📄"}</span><span class="name">${entry.name}</span>`;
    li.addEventListener("dblclick", async () => {
      if (entry.is_dir) {
        state.explorer.path = entry.path;
        await loadExplorer();
      } else if (ex.mode !== "dirs") {
        state.explorer.selected = entry.path;
        await confirmExplorer();
      }
    });
    li.addEventListener("click", () => {
      $$(".explorer-item").forEach((el) => el.classList.remove("active"));
      li.classList.add("active");
      if (ex.mode === "dirs") {
        state.explorer.selected = entry.is_dir ? entry.path : null;
      } else {
        state.explorer.selected = entry.is_dir ? null : entry.path;
      }
      $("#explorer-select").disabled = !state.explorer.selected;
    });
    list.appendChild(li);
  }
}

async function confirmExplorer() {
  const conf = browseTargets()[state.explorer.target];
  const selected = state.explorer.selected;
  if (!conf || !selected) return;
  $("#explorer").close();
  await conf.onSelect(selected);
  setStatus(t("msg.path_selected", { selected }), "ok");
}

async function uploadFile(kind, file) {
  const body = new FormData();
  body.append("file", file);
  const res = await fetch(`/api/upload?kind=${kind}`, { method: "POST", body });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || t("msg.upload_failed"));
  return data;
}

async function refreshJobs() {
  const data = await api("/api/jobs");
  const root = $("#job-list");
  root.innerHTML = "";
  if (!data.jobs.length) {
    root.innerHTML = `<li>${t("msg.no_jobs")}</li>`;
    return;
  }
  for (const job of data.jobs.slice(0, 12)) {
    const li = document.createElement("li");
    li.className = `job-item ${state.currentJobId === job.id ? "active" : ""}`;
    li.innerHTML = `<div><strong>${job.kind}</strong> <span class="badge ${job.status}">${job.status}</span><div class="meta">${job.id} · ${job.config_path}</div></div>`;
    li.addEventListener("click", () => selectJob(job.id));
    root.appendChild(li);
  }
}

function parseProgress(lines) {
  let best = null;
  for (let i = lines.length - 1; i >= 0; i -= 1) {
    const line = lines[i];
    const epoch = line.match(/Epoch\s+(\d+)\s*\/\s*(\d+)/i);
    const pct = line.match(/(\d+(?:\.\d+)?)\s*%/);
    const loss = line.match(/loss\s*=\s*([0-9.]+)/i);
    // Prefer batch/step fraction (e.g. 232/2700), not Epoch 1/2.
    const allFracs = [...line.matchAll(/(\d+)\s*\/\s*(\d+)/g)].map((m) => ({
      cur: Number(m[1]),
      tot: Number(m[2]) || 1,
      raw: m[0],
    }));
    let steps = null;
    if (allFracs.length) {
      const epCur = epoch ? Number(epoch[1]) : null;
      const epTot = epoch ? Number(epoch[2]) : null;
      const candidates = allFracs.filter(
        (f) => !(epCur != null && f.cur === epCur && f.tot === epTot)
      );
      steps = (candidates.length ? candidates : allFracs).reduce((a, b) =>
        b.tot > a.tot ? b : a
      );
    }
    if (!epoch && !pct && !steps) continue;

    const ep = epoch ? Number(epoch[1]) : null;
    const rawEps = epoch ? Number(epoch[2]) : null;
    const eps = rawEps != null && rawEps < 10000 ? rawEps : null;
    const stepPct =
      steps != null
        ? Math.min(100, (steps.cur / Math.max(steps.tot, 1)) * 100)
        : pct
          ? Number(pct[1])
          : null;
    let overallPct = null;
    if (ep != null && eps != null && stepPct != null) {
      overallPct = Math.min(100, (((ep - 1) + stepPct / 100) / Math.max(eps, 1)) * 100);
    } else if (stepPct != null && !epoch) {
      overallPct = stepPct;
    } else if (ep != null && eps != null) {
      overallPct = Math.min(100, ((ep - 1) / Math.max(eps, 1)) * 100);
    }

    const epochDetail = [];
    if (ep != null && eps != null) epochDetail.push(`Epoch ${ep}/${eps}`);
    else if (ep != null) epochDetail.push(`Epoch ${ep}`);
    if (steps) epochDetail.push(`${steps.cur}/${steps.tot}`);
    if (loss) epochDetail.push(`loss ${loss[1]}`);

    const overallDetail = [];
    if (ep != null && eps != null) overallDetail.push(`Epoch ${ep}/${eps}`);
    else if (ep != null) overallDetail.push(`Epoch ${ep}`);
    if (steps && eps) {
      const done = (ep - 1) * steps.tot + steps.cur;
      const all = eps * steps.tot;
      overallDetail.push(`${done}/${all}`);
    }
    if (loss) overallDetail.push(`loss ${loss[1]}`);

    if (overallPct == null && stepPct == null) continue;
    best = {
      overall: {
        percent: Math.max(0, Math.min(100, overallPct ?? stepPct ?? 0)),
        detail: overallDetail.join(" · "),
      },
      epoch: {
        percent: Math.max(0, Math.min(100, stepPct ?? 0)),
        detail: epochDetail.join(" · "),
      },
    };
    break;
  }
  return best;
}

function fmtPct(n) {
  return `${n.toFixed(n >= 10 ? 0 : 1)}%`;
}

function updateProgress(lines, status) {
  const box = $("#progress-box");
  const parsed = parseProgress(lines);
  if (!parsed && status !== "completed") {
    box.hidden = true;
    return;
  }
  box.hidden = false;
  const done = status === "completed";
  const overall = done ? 100 : parsed?.overall?.percent ?? 0;
  const epoch = done ? 100 : parsed?.epoch?.percent ?? 0;
  $("#progress-label-overall").textContent = fmtPct(overall);
  $("#progress-fill-overall").style.width = `${overall}%`;
  $("#progress-detail-overall").textContent =
    parsed?.overall?.detail || (done ? t("msg.completed") : "");
  $("#progress-label-epoch").textContent = fmtPct(epoch);
  $("#progress-fill-epoch").style.width = `${epoch}%`;
  $("#progress-detail-epoch").textContent =
    parsed?.epoch?.detail || (done ? t("msg.completed") : "");
}

function isNearBottom(el, threshold = 80) {
  return el.scrollHeight - el.scrollTop - el.clientHeight <= threshold;
}

async function selectJob(jobId) {
  state.currentJobId = jobId;
  $("#btn-stop").disabled = false;
  $("#log-meta").textContent = t("msg.job_meta", { id: jobId });
  await refreshJobs();
  await pollLogs();
}

async function pollLogs() {
  if (!state.currentJobId) return;
  try {
    const data = await api(`/api/jobs/${state.currentJobId}/logs?tail=400`);
    const frame = $("#log-frame");
    const stick = isNearBottom(frame);
    $("#log-view").textContent = data.lines.join("\n") || t("msg.empty_log");
    $("#log-meta").textContent = t("msg.log_meta", {
      id: state.currentJobId,
      status: data.status,
      n: data.total_lines || data.lines.length,
    });
    updateProgress(data.lines, data.status);
    if (stick) frame.scrollTop = frame.scrollHeight;
    $("#btn-stop").disabled = data.status !== "running";
    if (data.status === "running" || data.status === "queued") {
      clearTimeout(state.pollTimer);
      state.pollTimer = setTimeout(pollLogs, 1500);
    } else {
      await refreshJobs();
      await refreshAssets();
    }
  } catch (err) {
    setStatus(err.message, "bad");
  }
}

async function saveConfigOnly() {
  try {
    const payload = { ...formPayload(), start_training: false };
    if (!payload.dataset_paths.length) throw new Error(t("msg.need_dataset"));
    const data = await api("/api/configs", { method: "POST", body: JSON.stringify(payload) });
    state.lastConfigPath = data.config_path;
    $("#btn-export").disabled = false;
    setStatus(t("msg.yaml_saved", { path: data.config_path }), "ok");
    updateSetupSummary();
    await refreshAssets();
  } catch (err) {
    setStatus(err.message, "bad");
  }
}

async function startTrain(e) {
  e?.preventDefault?.();
  try {
    const payload = { ...formPayload(), start_training: true };
    if (!payload.dataset_paths.length) throw new Error(t("msg.need_dataset"));
    switchTab("train");
    setStatus(t("msg.starting_train"));
    const data = await api("/api/jobs/train", { method: "POST", body: JSON.stringify(payload) });
    state.lastConfigPath = data.config_path;
    state.currentJobId = data.job?.id || null;
    $("#btn-export").disabled = false;
    setStatus(t("msg.train_started", { path: data.config_path }), "ok");
    updateSetupSummary();
    await refreshJobs();
    await pollLogs();
    $("#log-frame")?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  } catch (err) {
    setStatus(err.message, "bad");
  }
}

async function startExport() {
  try {
    syncExportFieldsFromUi();
    if (!state.lastConfigPath) throw new Error(t("msg.need_config_export"));
    const payload = { ...formPayload(), start_training: false };
    const saved = await api("/api/configs", { method: "POST", body: JSON.stringify(payload) });
    state.lastConfigPath = saved.config_path;
    switchTab("export");
    setStatus(t("msg.export_starting"));
    const data = await api("/api/jobs/export", {
      method: "POST",
      body: JSON.stringify({ config_path: state.lastConfigPath }),
    });
    state.currentJobId = data.job.id;
    setStatus(t("msg.export_started", { id: data.job.id }), "ok");
    switchTab("train");
    await refreshJobs();
    await pollLogs();
  } catch (err) {
    setStatus(err.message, "bad");
  }
}

async function stopJob() {
  if (!state.currentJobId) return;
  try {
    await api(`/api/jobs/${state.currentJobId}/stop`, { method: "POST" });
    setStatus(t("msg.job_stopped"), "ok");
    await refreshJobs();
    await pollLogs();
  } catch (err) {
    setStatus(err.message, "bad");
  }
}

function wire() {
  $("#btn-refresh").addEventListener("click", () => refreshAssets().catch((e) => setStatus(e.message, "bad")));
  $("#load-config").addEventListener("change", () => loadSelectedConfig().catch((e) => setStatus(e.message, "bad")));
  $("#local-model").addEventListener("change", (e) => {
    if (e.target.value) $("#base_model").value = `./${e.target.value}`.replace(/^\.\/\.\//, "./");
  });
  $("#project_name").addEventListener("change", () => {
    $("#output_dir").value = "";
    setExportPath("adapter_path", "");
    setExportPath("merged_path", "");
    setExportPath("gguf_filename", "");
    syncDerivedPaths();
  });
  $("#btn-save").addEventListener("click", saveConfigOnly);
  $("#btn-save-and-train")?.addEventListener("click", async () => {
    await saveConfigOnly();
    if (state.lastConfigPath) switchTab("train");
  });
  $("#btn-train").addEventListener("click", startTrain);
  $("#train-form").addEventListener("submit", (e) => e.preventDefault());
  $("#btn-export").addEventListener("click", startExport);
  $("#btn-stop").addEventListener("click", stopJob);
  $("#btn-goto-train")?.addEventListener("click", () => switchTab("train"));
  $("#btn-goto-setup")?.addEventListener("click", () => switchTab("setup"));
  $("#btn-goto-chat")?.addEventListener("click", () => switchTab("chat"));
  ["adapter_path_ui", "merged_path_ui", "gguf_filename_ui"].forEach((id) => {
    $(`#${id}`)?.addEventListener("change", syncExportFieldsFromUi);
  });
  ["project_name", "base_model", "output_dir"].forEach((id) => {
    $(`#${id}`)?.addEventListener("input", updateSetupSummary);
  });

  $$("[data-browse]").forEach((btn) => {
    btn.addEventListener("click", () => {
      openExplorer(btn.dataset.browse).catch((e) => setStatus(e.message, "bad"));
    });
  });
  $("#btn-validate-model")?.addEventListener("click", () => {
    validateBaseModel().catch((e) => setStatus(e.message, "bad"));
  });
  $("#btn-save-hf-token")?.addEventListener("click", () => {
    saveHfToken().catch((e) => setStatus(e.message, "bad"));
  });
  $("#btn-clear-hf-token")?.addEventListener("click", () => {
    clearHfToken().catch((e) => setStatus(e.message, "bad"));
  });

  $("#explorer-close").addEventListener("click", () => $("#explorer").close());
  $("#explorer-cancel").addEventListener("click", () => $("#explorer").close());
  $("#explorer-select").addEventListener("click", () => confirmExplorer().catch((e) => setStatus(e.message, "bad")));
  $("#explorer-up").addEventListener("click", async () => {
    if (state.explorer.parent == null) {
      state.explorer.path = ".";
    } else {
      state.explorer.path = state.explorer.parent || ".";
    }
    await loadExplorer();
  });

  $("#upload-config").addEventListener("change", async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    try {
      const data = await uploadFile("config", file);
      await refreshAssets();
      $("#load-config").value = data.path;
      await loadSelectedConfig();
      setStatus(t("msg.config_uploaded", { path: data.path }), "ok");
    } catch (err) {
      setStatus(err.message, "bad");
    } finally {
      e.target.value = "";
    }
  });

  $("#upload-dataset").addEventListener("change", async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    try {
      const data = await uploadFile("dataset", file);
      state.selectedDatasets.add(data.path);
      await refreshAssets();
      setStatus(t("msg.dataset_uploaded", { path: data.path }), "ok");
    } catch (err) {
      setStatus(err.message, "bad");
    } finally {
      e.target.value = "";
    }
  });

  $$(".tab").forEach((btn) => {
    btn.addEventListener("click", () => switchTab(btn.dataset.tab));
  });
  $("#btn-refresh-ollama").addEventListener("click", () => refreshOllama().catch((e) => setChatStatus(e.message, "bad")));
  $("#chat-gguf").addEventListener("change", (e) => {
    const opt = e.target.selectedOptions?.[0];
    if (opt?.dataset?.suggested) $("#chat-model-name").value = opt.dataset.suggested;
  });
  $("#btn-register-gguf").addEventListener("click", () => registerGguf().catch((e) => setChatStatus(e.message, "bad")));
  $("#btn-clear-chat").addEventListener("click", () => {
    state.chatMessages = [];
    renderChatMessages();
    setChatStatus("");
  });
  $("#chat-form").addEventListener("submit", sendChat);

  $("#lang-select")?.addEventListener("change", (e) => {
    window.FT_I18N.setLocale(e.target.value);
  });
  document.addEventListener("ft:locale", onLocaleChange);
}

function switchTab(tab) {
  state.tab = tab || "setup";
  $$(".tab").forEach((el) => el.classList.toggle("active", el.dataset.tab === state.tab));
  const views = {
    setup: $("#view-setup"),
    train: $("#view-train"),
    export: $("#view-export"),
    chat: $("#view-chat"),
  };
  Object.entries(views).forEach(([name, el]) => {
    if (!el) return;
    el.hidden = name !== state.tab;
  });
  if (state.tab === "export") syncExportUiFromHidden();
  if (state.tab === "train" || state.tab === "setup") updateSetupSummary();
  if (state.tab === "chat") refreshOllama().catch((e) => setChatStatus(e.message, "bad"));
}

function setChatStatus(msg, kind = "") {
  const el = $("#chat-status");
  el.textContent = msg || "";
  el.className = `status ${kind}`.trim();
}

function onLocaleChange() {
  if (state.assets) renderAssets(state.assets);
  updateSetupSummary();
  renderChatMessages();
  refreshJobs().catch(() => {});
  if (!state.currentJobId) {
    $("#log-meta").textContent = t("train.no_job");
    $("#log-view").textContent = t("train.no_job_log");
  }
  const stEl = $("#ollama-status");
  if (stEl && !stEl.classList.contains("ok") && !stEl.classList.contains("bad")) {
    stEl.textContent = t("chat.checking");
  }
  if (state.tab === "chat") refreshOllama().catch(() => {});
}

function renderChatMessages() {
  const root = $("#chat-messages");
  root.innerHTML = "";
  if (!state.chatMessages.length) {
    root.innerHTML = `<p class="hint" style="color:#9bb5a8">${t("msg.chat_empty")}</p>`;
    return;
  }
  for (const m of state.chatMessages) {
    const div = document.createElement("div");
    div.className = `chat-bubble ${m.role}`;
    div.innerHTML = `<span class="role">${m.role}</span>${escapeHtml(m.content)}`;
    root.appendChild(div);
  }
  root.scrollTop = root.scrollHeight;
}

function escapeHtml(text) {
  return String(text)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

async function refreshOllama() {
  const stEl = $("#ollama-status");
  try {
    const st = await api("/api/ollama/status");
    if (!st.ok) {
      stEl.textContent = t("msg.ollama_offline", { host: st.host });
      stEl.className = "status bad";
      return;
    }
    stEl.textContent = t("msg.ollama_online", { host: st.host });
    stEl.className = "status ok";
    const data = await api("/api/ollama/models");
    if (!$("#chat-system").value) $("#chat-system").value = data.default_system || "";

    const modelSel = $("#chat-model");
    const current = modelSel.value;
    modelSel.innerHTML = `<option value="">${t("msg.dash")}</option>`;
    for (const m of data.models || []) {
      const opt = document.createElement("option");
      opt.value = m.name;
      opt.textContent = m.name;
      modelSel.appendChild(opt);
    }
    if (current && [...modelSel.options].some((o) => o.value === current)) modelSel.value = current;
    else if ((data.models || []).length) modelSel.value = data.models[0].name;

    const ggufSel = $("#chat-gguf");
    ggufSel.innerHTML = `<option value="">${t("msg.dash")}</option>`;
    for (const g of data.ggufs || []) {
      const opt = document.createElement("option");
      opt.value = g.path;
      opt.dataset.suggested = g.suggested_model;
      opt.textContent = `${g.name} (${Math.round((g.size_bytes || 0) / 1024 / 1024)} MB)`;
      ggufSel.appendChild(opt);
    }
  } catch (err) {
    stEl.textContent = err.message;
    stEl.className = "status bad";
  }
}

async function registerGguf() {
  const gguf_path = $("#chat-gguf").value;
  if (!gguf_path) throw new Error(t("msg.need_gguf"));
  const model_name = $("#chat-model-name").value.trim() || undefined;
  setChatStatus(t("msg.registering"));
  $("#btn-register-gguf").disabled = true;
  try {
    const data = await api("/api/ollama/register", {
      method: "POST",
      body: JSON.stringify({
        gguf_path,
        model_name,
        system_prompt: $("#chat-system").value,
      }),
    });
    setChatStatus(t("msg.model_created", { model: data.model }), "ok");
    await refreshOllama();
    $("#chat-model").value = data.model;
  } finally {
    $("#btn-register-gguf").disabled = false;
  }
}

async function sendChat(e) {
  e.preventDefault();
  const model = $("#chat-model").value;
  const content = $("#chat-input").value.trim();
  if (!model) {
    setChatStatus(t("msg.need_ollama_model"), "bad");
    return;
  }
  if (!content) return;

  state.chatMessages.push({ role: "user", content });
  $("#chat-input").value = "";
  renderChatMessages();
  setChatStatus(t("msg.generating"));
  $("#btn-send-chat").disabled = true;
  try {
    const data = await api("/api/ollama/chat", {
      method: "POST",
      body: JSON.stringify({
        model,
        system_prompt: $("#chat-system").value,
        messages: state.chatMessages,
      }),
    });
    state.chatMessages.push({
      role: data.message?.role || "assistant",
      content: data.message?.content || "",
    });
    renderChatMessages();
    setChatStatus(t("msg.ok"), "ok");
  } catch (err) {
    state.chatMessages.pop();
    renderChatMessages();
    setChatStatus(err.message, "bad");
  } finally {
    $("#btn-send-chat").disabled = false;
  }
}

async function boot() {
  window.FT_I18N?.init();
  wire();
  syncDerivedPaths();
  $("#log-meta").textContent = t("train.no_job");
  $("#log-view").textContent = t("train.no_job_log");
  $("#ollama-status").textContent = t("chat.checking");
  await refreshAssets();
  await refreshHfTokenStatus();
  await refreshJobs();
  const preferred =
    state.assets?.preferred_config ||
    (state.assets?.configs || []).find((c) => c.path.includes("default"))?.path ||
    (state.assets?.configs || [])[0]?.path;
  if (preferred) {
    $("#load-config").value = preferred;
    await loadSelectedConfig();
  }
  const jobs = await api("/api/jobs");
  const active = (jobs.jobs || []).find((j) => j.status === "running" || j.status === "queued");
  if (active) {
    await selectJob(active.id);
    switchTab("train");
  } else {
    if ((jobs.jobs || []).length) await selectJob(jobs.jobs[0].id);
    switchTab("setup");
  }
  updateSetupSummary();
}

boot().catch((err) => setStatus(err.message, "bad"));
