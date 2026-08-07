const state = {
  config: null,
  runs: [],
  currentRun: null,
  currentRunId: window.location.hash.slice(1),
  selectedShotId: null,
  toastTimer: null,
};

const elements = Object.fromEntries(
  [
    "api-indicator", "api-label", "run-list", "run-count", "run-search", "page-title",
    "empty-state", "run-workspace", "run-status", "run-id", "run-title", "run-request",
    "resume-button", "download-button", "video-player", "video-placeholder", "preview-kind",
    "meta-ratio", "meta-resolution", "meta-duration", "meta-style", "updated-time", "stage-label",
    "stage-percent", "progress-bar", "shot-progress", "take-progress", "platforms-value",
    "review-value", "run-error", "export-list", "shot-count", "shot-list", "create-dialog",
    "create-form", "ratio-options", "resolution-options", "style-select", "music-select",
    "platform-options", "form-error", "submit-run", "toast",
  ].map((id) => [id, document.getElementById(id)]),
);

const statusLabels = {
  created: "已创建",
  running: "制作中",
  succeeded: "已完成",
  failed: "已失败",
  interrupted: "待继续",
  pending: "等待中",
  success: "已通过",
};

const stageLabels = {
  initialized: "准备工作区",
  storyboard_ready: "分镜已确认",
  generating: "生成镜头",
  generating_shots: "生成镜头",
  postprocessing: "后期处理",
  completed: "制作完成",
};

const platformLabels = {
  youtube: "YouTube",
  tiktok: "TikTok",
  bilibili: "Bilibili",
  instagram_reels: "Instagram Reels",
  instagram_feed: "Instagram Feed",
};

function createElement(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function icon(name) {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("aria-hidden", "true");
  const use = document.createElementNS("http://www.w3.org/2000/svg", "use");
  use.setAttribute("href", `#icon-${name}`);
  svg.append(use);
  return svg;
}

async function requestJSON(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.error || `请求失败 (${response.status})`);
  return body;
}

function formatTime(value) {
  if (!value) return "";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "";
  return new Intl.DateTimeFormat("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  }).format(date);
}

function showToast(message) {
  elements.toast.textContent = message;
  elements.toast.classList.add("is-visible");
  window.clearTimeout(state.toastTimer);
  state.toastTimer = window.setTimeout(() => elements.toast.classList.remove("is-visible"), 2800);
}

function renderRunList() {
  const query = elements["run-search"].value.trim().toLowerCase();
  const runs = state.runs.filter((run) => `${run.request} ${run.run_id}`.toLowerCase().includes(query));
  elements["run-list"].replaceChildren();
  elements["run-count"].textContent = String(state.runs.length);

  for (const run of runs) {
    const button = createElement("button", `run-item${run.run_id === state.currentRunId ? " is-active" : ""}`);
    button.type = "button";
    button.dataset.runId = run.run_id;
    const dot = createElement("span", `status-dot ${run.active ? "running" : run.status}`);
    const copy = createElement("span", "run-item-copy");
    copy.append(
      createElement("strong", "", run.request || "未命名任务"),
      createElement("span", "tabular", formatTime(run.updated_at)),
    );
    button.append(dot, copy, icon("arrow"));
    button.addEventListener("click", () => selectRun(run.run_id));
    elements["run-list"].append(button);
  }
}

function stageProgress(run) {
  if (run.status === "succeeded") return 100;
  const total = run.progress.total;
  const complete = run.progress.completed;
  if (total > 0) return Math.min(90, 20 + Math.round((complete / total) * 65));
  return run.stage === "storyboard_ready" ? 18 : run.active ? 8 : 0;
}

function setVideo(url, label, poster = "") {
  const currentSource = elements["video-player"].getAttribute("src");
  if (url) {
    if (currentSource !== url) {
      elements["video-player"].src = url;
      elements["video-player"].poster = poster || "";
      elements["video-player"].load();
    }
    elements["video-player"].hidden = false;
    elements["video-placeholder"].hidden = true;
    elements["preview-kind"].textContent = label;
  } else {
    elements["video-player"].removeAttribute("src");
    elements["video-player"].removeAttribute("poster");
    elements["video-player"].hidden = true;
    elements["video-placeholder"].hidden = false;
    elements["preview-kind"].textContent = "等待素材";
  }
}

function selectShot(shot) {
  if (!shot.video_url) return;
  state.selectedShotId = shot.shot_id;
  setVideo(shot.video_url, `镜头 ${String(shot.shot_id).padStart(2, "0")}`, shot.poster_url);
  renderShots(state.currentRun.shots);
}

function renderShots(shots) {
  elements["shot-list"].replaceChildren();
  elements["shot-count"].textContent = `${shots.length} 个镜头`;

  for (const shot of shots) {
    const button = createElement(
      "button",
      `shot-item${shot.shot_id === state.selectedShotId ? " is-selected" : ""}`,
    );
    button.type = "button";
    button.disabled = !shot.video_url;

    const thumb = createElement("span", "shot-thumb tabular", String(shot.shot_id).padStart(2, "0"));
    if (shot.poster_url) {
      const image = document.createElement("img");
      image.src = shot.poster_url;
      image.alt = `镜头 ${shot.shot_id}`;
      image.loading = "lazy";
      thumb.replaceChildren(image);
    }
    if (shot.status === "success") {
      const mark = createElement("span", "shot-state-mark");
      mark.append(icon("check"));
      thumb.append(mark);
    }

    const copy = createElement("span", "shot-item-copy");
    const meta = createElement("span", "shot-item-meta");
    meta.append(
      createElement("span", "", `Shot ${String(shot.shot_id).padStart(2, "0")}`),
      createElement("span", "tabular", shot.duration ? `${shot.duration}s` : statusLabels[shot.status] || shot.status),
    );
    copy.append(
      meta,
      createElement("strong", "", shot.scene_description || shot.primary_action || "等待分镜内容"),
      createElement("span", "", `${statusLabels[shot.status] || shot.status} · ${shot.quality_score || 0} 分`),
    );
    button.append(thumb, copy);
    if (shot.video_url) button.addEventListener("click", () => selectShot(shot));
    elements["shot-list"].append(button);
  }
}

function renderExports(exports) {
  elements["export-list"].replaceChildren();
  for (const [platform, url] of Object.entries(exports)) {
    const link = createElement("a", "export-link");
    link.href = url;
    link.download = "";
    link.append(createElement("span", "", platformLabels[platform] || platform), icon("download"));
    elements["export-list"].append(link);
  }
}

function renderRun(run) {
  state.currentRun = run;
  elements["empty-state"].hidden = true;
  elements["run-workspace"].hidden = false;
  elements["page-title"].textContent = run.title || "制作详情";
  elements["run-status"].textContent = run.active ? "制作中" : statusLabels[run.status] || run.status;
  elements["run-status"].className = `status-badge ${run.active ? "running" : run.status}`;
  elements["run-id"].textContent = run.run_id;
  elements["run-title"].textContent = run.title || run.request || "未命名任务";
  elements["run-request"].textContent = run.request;

  elements["resume-button"].hidden = !run.can_resume;
  elements["download-button"].hidden = !run.assets.final_url;
  elements["download-button"].href = run.assets.final_url || "#";
  elements["download-button"].setAttribute("download", `${run.title || run.run_id}.mp4`);

  const defaultShot = run.shots.find((shot) => shot.video_url);
  if (!state.selectedShotId || !run.shots.some((shot) => shot.shot_id === state.selectedShotId)) {
    state.selectedShotId = null;
    setVideo(run.assets.final_url || defaultShot?.video_url, run.assets.final_url ? "最终成片" : "最新镜头", run.assets.poster_url);
  }

  elements["meta-ratio"].textContent = run.options.aspect_ratio;
  elements["meta-resolution"].textContent = run.options.resolution;
  elements["meta-duration"].textContent = run.total_duration ? `${run.total_duration}s` : "-";
  elements["meta-style"].textContent = run.options.style;
  elements["updated-time"].textContent = formatTime(run.updated_at);
  elements["stage-label"].textContent = stageLabels[run.stage] || run.stage;

  const percent = stageProgress(run);
  elements["stage-percent"].textContent = `${percent}%`;
  elements["progress-bar"].style.width = `${percent}%`;
  elements["shot-progress"].textContent = `${run.progress.completed} / ${run.progress.total}`;
  elements["take-progress"].textContent = `${run.budget.used} / ${run.budget.limit ?? "-"}`;
  elements["platforms-value"].textContent = run.options.platforms.map((item) => platformLabels[item] || item).join(", ") || "-";
  elements["review-value"].textContent = state.config?.semantic_review_enabled ? "已启用" : "仅技术验收";

  elements["run-error"].hidden = !run.error;
  elements["run-error"].querySelector("p").textContent = run.error || "";
  renderExports(run.assets.exports);
  renderShots(run.shots);
}

async function loadRuns({ keepSelection = true } = {}) {
  const payload = await requestJSON("/api/runs");
  state.runs = payload.runs;
  if (!keepSelection || !state.currentRunId) state.currentRunId = state.runs[0]?.run_id || "";
  renderRunList();
  if (state.currentRunId) await loadRun(state.currentRunId);
  else {
    elements["empty-state"].hidden = false;
    elements["run-workspace"].hidden = true;
  }
}

async function loadRun(runId) {
  const run = await requestJSON(`/api/runs/${encodeURIComponent(runId)}`);
  renderRun(run);
  const match = state.runs.find((item) => item.run_id === runId);
  if (match) Object.assign(match, run);
  renderRunList();
}

async function selectRun(runId) {
  state.currentRunId = runId;
  state.selectedShotId = null;
  window.location.hash = runId;
  renderRunList();
  try {
    await loadRun(runId);
  } catch (error) {
    showToast(error.message);
  }
}

function addSegmentedOptions(container, name, values, selected) {
  container.replaceChildren();
  for (const value of values) {
    const label = createElement("label", "segmented-option");
    const input = document.createElement("input");
    input.type = "radio";
    input.name = name;
    input.value = value;
    input.checked = value === selected;
    label.append(input, createElement("span", "", value));
    container.append(label);
  }
}

function renderFormOptions(config) {
  addSegmentedOptions(elements["ratio-options"], "aspect_ratio", config.aspect_ratios, "16:9");
  addSegmentedOptions(elements["resolution-options"], "resolution", config.resolutions, "480p");

  elements["style-select"].replaceChildren();
  for (const style of config.styles) {
    const option = document.createElement("option");
    option.value = style;
    option.textContent = style;
    elements["style-select"].append(option);
  }

  for (const music of config.music) {
    const option = document.createElement("option");
    option.value = music.value;
    option.textContent = music.label;
    elements["music-select"].append(option);
  }

  elements["platform-options"].replaceChildren();
  for (const platform of config.platforms) {
    const label = createElement("label", "check-option");
    const input = document.createElement("input");
    input.type = "checkbox";
    input.name = "platforms";
    input.value = platform;
    input.checked = ["youtube", "tiktok"].includes(platform);
    label.append(input, createElement("span", "", platformLabels[platform] || platform));
    elements["platform-options"].append(label);
  }
}

function openCreateDialog() {
  elements["form-error"].hidden = true;
  elements["create-dialog"].showModal();
  window.setTimeout(() => elements["create-form"].elements.request.focus(), 0);
}

function closeCreateDialog() {
  elements["create-dialog"].close();
}

async function submitCreate(event) {
  event.preventDefault();
  const form = new FormData(elements["create-form"]);
  const payload = {
    request: form.get("request"),
    aspect_ratio: form.get("aspect_ratio"),
    resolution: form.get("resolution"),
    style: form.get("style"),
    music: form.get("music"),
    paid_take_budget: form.get("paid_take_budget"),
    platforms: form.getAll("platforms"),
  };

  elements["submit-run"].disabled = true;
  elements["submit-run"].style.opacity = "0.6";
  elements["form-error"].hidden = true;
  try {
    const run = await requestJSON("/api/runs", { method: "POST", body: JSON.stringify(payload) });
    closeCreateDialog();
    elements["create-form"].reset();
    renderFormOptions(state.config);
    await loadRuns({ keepSelection: true });
    await selectRun(run.run_id);
    showToast("任务已开始");
  } catch (error) {
    elements["form-error"].textContent = error.message;
    elements["form-error"].hidden = false;
  } finally {
    elements["submit-run"].disabled = false;
    elements["submit-run"].style.opacity = "";
  }
}

async function resumeCurrentRun() {
  if (!state.currentRunId) return;
  elements["resume-button"].disabled = true;
  try {
    await requestJSON(`/api/runs/${encodeURIComponent(state.currentRunId)}/resume`, { method: "POST" });
    await loadRun(state.currentRunId);
    showToast("任务已继续");
  } catch (error) {
    showToast(error.message);
  } finally {
    elements["resume-button"].disabled = false;
  }
}

async function refresh() {
  try {
    await loadRuns({ keepSelection: true });
  } catch (error) {
    showToast(error.message);
  }
}

async function init() {
  try {
    state.config = await requestJSON("/api/config");
    elements["api-indicator"].classList.add(state.config.api_ready ? "ready" : "");
    elements["api-label"].textContent = state.config.api_ready ? "生成服务已就绪" : "尚未配置 API Key";
    renderFormOptions(state.config);
    await loadRuns({ keepSelection: Boolean(state.currentRunId) });
  } catch (error) {
    showToast(error.message);
  }

  window.setInterval(() => {
    if (!document.hidden && state.currentRunId) loadRun(state.currentRunId).catch(() => {});
  }, 3000);
}

document.getElementById("new-run-button").addEventListener("click", openCreateDialog);
document.getElementById("compact-new-button").addEventListener("click", openCreateDialog);
document.getElementById("empty-new-button").addEventListener("click", openCreateDialog);
document.getElementById("close-dialog").addEventListener("click", closeCreateDialog);
document.getElementById("cancel-dialog").addEventListener("click", closeCreateDialog);
document.getElementById("refresh-button").addEventListener("click", refresh);
elements["resume-button"].addEventListener("click", resumeCurrentRun);
elements["run-search"].addEventListener("input", renderRunList);
elements["create-form"].addEventListener("submit", submitCreate);
elements["create-dialog"].addEventListener("click", (event) => {
  if (event.target === elements["create-dialog"]) closeCreateDialog();
});
window.addEventListener("hashchange", () => {
  const runId = window.location.hash.slice(1);
  if (runId && runId !== state.currentRunId) selectRun(runId);
});

init();
