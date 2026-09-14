"use strict";

const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const maskCanvas = document.createElement("canvas");
const maskCtx = maskCanvas.getContext("2d");

const state = {
  project: null,
  frameIndex: 0,
  annotation: null,
  operations: [],
  redo: [],
  tool: "polygon",
  classId: 1,
  polygon: [],
  stroke: null,
  dirty: false,
  saving: false,
  queuedSave: false,
  savePromise: null,
  editGeneration: 0,
  image: new Image(),
  imageReady: false,
  alpha: 0.48,
  zoom: 1.0,
  rawOnly: false,
};

const shortcutClass = {q: 1, a: 2, w: 3, s: 4, e: 5, d: 6, o: 7, u: 8, x: 0};

function status(message, kind = "") {
  const el = document.getElementById("statusBar");
  el.textContent = message;
  el.className = kind;
}

async function api(url, options = {}) {
  const response = await fetch(url, options);
  const payload = await response.json();
  if (!response.ok || payload.ok === false) throw new Error(payload.error || `HTTP ${response.status}`);
  return payload;
}

function opId() {
  return (crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random()}`);
}

function clone(value) {
  return window.structuredClone ? structuredClone(value) : JSON.parse(JSON.stringify(value));
}

function frame() { return state.project.frames[state.frameIndex]; }

function setDirty(value = true) {
  state.dirty = value;
  if (value) state.editGeneration += 1;
  status(value ? "有未保存修改…" : `已保存 revision ${state.annotation?.revision ?? 0}`, value ? "" : "ok");
  if (value) {
    clearTimeout(setDirty.timer);
    setDirty.timer = setTimeout(() => save(), 650);
  }
}

function buildClasses() {
  const root = document.getElementById("classButtons");
  root.replaceChildren();
  for (const item of state.project.classes) {
    const button = document.createElement("button");
    button.className = "classButton";
    button.dataset.id = item.id;
    button.innerHTML = `<span class="swatch" style="background:${item.color}"></span><span>${item.name_zh}<br><small>${item.symbol}</small></span><kbd>${item.hotkey}</kbd>`;
    button.addEventListener("click", () => selectClass(item.id));
    root.appendChild(button);
  }
  selectClass(state.classId);
}

function selectClass(id) {
  state.classId = Number(id);
  for (const button of document.querySelectorAll(".classButton")) {
    button.classList.toggle("active", Number(button.dataset.id) === state.classId);
  }
  const item = state.project.classes.find(value => value.id === state.classId);
  status(`当前类别：${item.name_zh} / ${item.symbol}`);
}

function selectTool(tool) {
  state.tool = tool;
  state.polygon = [];
  state.stroke = null;
  document.getElementById("polygonTool").classList.toggle("active", tool === "polygon");
  document.getElementById("brushTool").classList.toggle("active", tool === "brush");
  render();
}

function drawOperation(target, op) {
  const item = state.project.classes.find(value => value.id === op.class_id);
  target.save();
  if (op.class_id === 0) {
    target.globalCompositeOperation = "destination-out";
    target.fillStyle = "#000";
    target.strokeStyle = "#000";
  } else {
    target.globalCompositeOperation = "source-over";
    target.fillStyle = item.color;
    target.strokeStyle = item.color;
  }
  if (op.kind === "polygon" && op.points.length >= 3) {
    target.beginPath();
    target.moveTo(op.points[0][0], op.points[0][1]);
    for (const p of op.points.slice(1)) target.lineTo(p[0], p[1]);
    target.closePath();
    target.fill();
  } else if (op.kind === "brush" && op.points.length) {
    target.lineWidth = op.radius * 2;
    target.lineCap = "round";
    target.lineJoin = "round";
    target.beginPath();
    target.moveTo(op.points[0][0], op.points[0][1]);
    for (const p of op.points.slice(1)) target.lineTo(p[0], p[1]);
    if (op.points.length === 1) target.lineTo(op.points[0][0] + 0.01, op.points[0][1]);
    target.stroke();
  }
  target.restore();
}

function rebuildMask() {
  maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
  for (const op of state.operations) drawOperation(maskCtx, op);
  if (state.stroke) drawOperation(maskCtx, state.stroke);
}

function render() {
  if (!state.imageReady) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(state.image, 0, 0);
  rebuildMask();
  if (!state.rawOnly) {
    ctx.save();
    ctx.globalAlpha = state.alpha;
    ctx.drawImage(maskCanvas, 0, 0);
    ctx.restore();
  }
  if (state.polygon.length) {
    ctx.save();
    ctx.strokeStyle = "#ffffff";
    ctx.fillStyle = "#ffffff";
    ctx.lineWidth = Math.max(1, 2 / state.zoom);
    ctx.beginPath();
    ctx.moveTo(state.polygon[0][0], state.polygon[0][1]);
    for (const p of state.polygon.slice(1)) ctx.lineTo(p[0], p[1]);
    ctx.stroke();
    for (const p of state.polygon) {
      ctx.beginPath(); ctx.arc(p[0], p[1], Math.max(2, 4 / state.zoom), 0, Math.PI * 2); ctx.fill();
    }
    ctx.restore();
  }
}

function pointFromEvent(event) {
  const rect = canvas.getBoundingClientRect();
  const x = Math.max(0, Math.min(canvas.width - 1, (event.clientX - rect.left) * canvas.width / rect.width));
  const y = Math.max(0, Math.min(canvas.height - 1, (event.clientY - rect.top) * canvas.height / rect.height));
  return [Number(x.toFixed(2)), Number(y.toFixed(2))];
}

function commit(op) {
  state.operations.push(op);
  state.redo = [];
  setDirty(true);
  render();
}

function finishPolygon() {
  const points = state.polygon;
  state.polygon = [];
  if (points.length >= 3) commit({op_id: opId(), kind: "polygon", class_id: state.classId, points});
  else render();
}

function undo() {
  if (state.polygon.length) { state.polygon.pop(); render(); return; }
  const op = state.operations.pop();
  if (op) { state.redo.push(op); setDirty(true); render(); }
}

function redo() {
  const op = state.redo.pop();
  if (op) { state.operations.push(op); setDirty(true); render(); }
}

async function save() {
  if (!state.annotation || !state.dirty) return;
  if (state.saving) { state.queuedSave = true; return state.savePromise; }
  state.saving = true;
  state.savePromise = (async () => {
    try {
      do {
        state.queuedSave = false;
        if (!state.dirty) break;
        clearTimeout(setDirty.timer);
        const currentFrame = frame();
        const generation = state.editGeneration;
        const payload = {
          revision: state.annotation.revision,
          complete: document.getElementById("completeCheck").checked,
          operations: clone(state.operations),
        };
        const result = await api(currentFrame.annotation_url, {
          method: "PUT", headers: {"Content-Type": "application/json"}, body: JSON.stringify(payload),
        });
        state.annotation = result.annotation;
        currentFrame.complete = result.annotation.complete;
        currentFrame.revision = result.annotation.revision;
        if (state.editGeneration === generation) {
          state.operations = clone(result.annotation.operations);
          state.dirty = false;
          refreshFrameSelect();
          status(`已保存 ${currentFrame.frame_key} · revision ${result.annotation.revision}`, "ok");
        } else {
          state.queuedSave = true;
          status("保存期间产生了新修改，正在继续保存…");
        }
      } while (state.queuedSave || state.dirty);
    } catch (error) {
      status(`保存失败：${error.message}。请勿关闭页面。`, "error");
      throw error;
    } finally {
      state.saving = false;
      state.savePromise = null;
    }
  })();
  return state.savePromise;
}

function refreshFrameSelect() {
  const select = document.getElementById("frameSelect");
  select.replaceChildren();
  state.project.frames.forEach((item, index) => {
    const option = document.createElement("option");
    option.value = index;
    option.textContent = `${item.complete ? "✓" : "○"} ${index + 1}/${state.project.frames.length} · source #${item.source_decoded_index} · ${item.source_pts_time}s`;
    option.selected = index === state.frameIndex;
    select.appendChild(option);
  });
}

async function loadFrame(index) {
  if (state.annotation && state.dirty) await save();
  state.frameIndex = Math.max(0, Math.min(state.project.frames.length - 1, Number(index)));
  const item = frame();
  status(`读取 ${item.frame_key}…`);
  const annotationPromise = api(item.annotation_url);
  state.imageReady = false;
  state.image = new Image();
  const imagePromise = new Promise((resolve, reject) => {
    state.image.onload = resolve;
    state.image.onerror = () => reject(new Error("图像载入失败"));
  });
  state.image.src = `${item.image_url}?sha=${item.image_sha256}`;
  const annotation = await annotationPromise;
  await imagePromise;
  state.annotation = annotation;
  state.operations = clone(annotation.operations);
  state.redo = [];
  state.polygon = [];
  state.stroke = null;
  state.dirty = false;
  canvas.width = item.image_width;
  canvas.height = item.image_height;
  maskCanvas.width = item.image_width;
  maskCanvas.height = item.image_height;
  state.imageReady = true;
  document.getElementById("completeCheck").checked = annotation.complete;
  document.getElementById("frameMeta").textContent = `sample: ${item.sample_id}\nPTS: ${item.source_pts_time}s\ndecoded index: ${item.source_decoded_index}\n原图: ${item.image_width}×${item.image_height}`;
  document.getElementById("prevButton").disabled = state.frameIndex === 0;
  document.getElementById("nextButton").disabled = state.frameIndex === state.project.frames.length - 1;
  refreshFrameSelect();
  applyZoom();
  render();
  status(`已载入 ${item.frame_key} · ${annotation.operations.length} 个操作 · revision ${annotation.revision}`, "ok");
}

function applyZoom() {
  canvas.style.width = `${canvas.width * state.zoom}px`;
  canvas.style.height = `${canvas.height * state.zoom}px`;
  document.getElementById("zoomValue").textContent = Math.round(state.zoom * 100);
}

async function action(path, label) {
  try {
    await save();
    status(`${label}中…`);
    const result = await api(path, {method: "POST"});
    const detail = result.result;
    if (path.endsWith("validate")) {
      status(`校验 ${detail.status} · ${detail.errors.length} errors · ${detail.warnings.length} warnings`, detail.status === "PASS" ? "ok" : "error");
    } else {
      status(`已导出 ${detail.frame_count} 帧到 output_dir/export`, "ok");
    }
  } catch (error) { status(`${label}失败：${error.message}`, "error"); }
}

canvas.addEventListener("click", event => {
  if (state.tool !== "polygon" || event.detail > 1) return;
  state.polygon.push(pointFromEvent(event));
  render();
});
canvas.addEventListener("dblclick", event => {
  if (state.tool !== "polygon") return;
  event.preventDefault();
  const point = pointFromEvent(event);
  const last = state.polygon[state.polygon.length - 1];
  if (!last || Math.hypot(last[0] - point[0], last[1] - point[1]) > 2) state.polygon.push(point);
  finishPolygon();
});
canvas.addEventListener("pointerdown", event => {
  if (state.tool !== "brush") return;
  canvas.setPointerCapture(event.pointerId);
  state.stroke = {op_id: opId(), kind: "brush", class_id: state.classId, radius: Number(document.getElementById("brushRadius").value), points: [pointFromEvent(event)]};
  render();
});
canvas.addEventListener("pointermove", event => {
  if (!state.stroke) return;
  const point = pointFromEvent(event);
  const last = state.stroke.points[state.stroke.points.length - 1];
  if (Math.hypot(last[0] - point[0], last[1] - point[1]) >= 0.8) state.stroke.points.push(point);
  render();
});
function finishStroke() {
  if (!state.stroke) return;
  const op = state.stroke;
  state.stroke = null;
  commit(op);
}
canvas.addEventListener("pointerup", finishStroke);
canvas.addEventListener("pointercancel", finishStroke);

document.getElementById("saveButton").onclick = () => save();
document.getElementById("validateButton").onclick = () => action("/api/validate", "校验");
document.getElementById("exportButton").onclick = () => action("/api/export", "导出");
document.getElementById("prevButton").onclick = () => loadFrame(state.frameIndex - 1);
document.getElementById("nextButton").onclick = () => loadFrame(state.frameIndex + 1);
document.getElementById("frameSelect").onchange = event => loadFrame(event.target.value);
document.getElementById("polygonTool").onclick = () => selectTool("polygon");
document.getElementById("brushTool").onclick = () => selectTool("brush");
document.getElementById("undoButton").onclick = undo;
document.getElementById("redoButton").onclick = redo;
document.getElementById("completeCheck").onchange = () => setDirty(true);
document.getElementById("brushRadius").oninput = event => { document.getElementById("brushValue").textContent = event.target.value; };
document.getElementById("alphaSlider").oninput = event => { state.alpha = Number(event.target.value) / 100; document.getElementById("alphaValue").textContent = event.target.value; render(); };
document.getElementById("zoomSlider").oninput = event => { state.zoom = Number(event.target.value) / 100; applyZoom(); };
document.getElementById("rawOnlyCheck").onchange = event => { state.rawOnly = event.target.checked; render(); };

window.addEventListener("keydown", event => {
  if (event.target.matches("input,select,textarea")) return;
  const key = event.key.toLowerCase();
  if ((event.ctrlKey || event.metaKey) && key === "s") { event.preventDefault(); save(); return; }
  if ((event.ctrlKey || event.metaKey) && key === "z") { event.preventDefault(); event.shiftKey ? redo() : undo(); return; }
  if ((event.ctrlKey || event.metaKey) && key === "y") { event.preventDefault(); redo(); return; }
  if (/^[0-8]$/.test(key)) { selectClass(Number(key)); return; }
  if (shortcutClass[key] !== undefined) { selectClass(shortcutClass[key]); return; }
  if (key === "p") selectTool("polygon");
  else if (key === "b") selectTool("brush");
  else if (key === "enter") finishPolygon();
  else if (key === "escape") { state.polygon = []; state.stroke = null; render(); }
  else if (key === "arrowleft") loadFrame(state.frameIndex - 1);
  else if (key === "arrowright") loadFrame(state.frameIndex + 1);
  else if (key === "c") { const check = document.getElementById("completeCheck"); check.checked = !check.checked; setDirty(true); }
  else if (key === "tab") { event.preventDefault(); const check = document.getElementById("rawOnlyCheck"); check.checked = !check.checked; state.rawOnly = check.checked; render(); }
  else if (key === "[") { const input = document.getElementById("brushRadius"); input.value = Math.max(1, Number(input.value) - 2); input.oninput({target: input}); }
  else if (key === "]") { const input = document.getElementById("brushRadius"); input.value = Math.min(120, Number(input.value) + 2); input.oninput({target: input}); }
});

window.addEventListener("beforeunload", event => {
  if (state.dirty) { event.preventDefault(); event.returnValue = ""; }
});

(async function init() {
  try {
    const payload = await api("/api/project");
    state.project = payload;
    document.getElementById("sessionLine").textContent = `${payload.session_id} · ${payload.frames.length} 帧 · ${payload.sampling.method}`;
    buildClasses();
    refreshFrameSelect();
    await loadFrame(0);
  } catch (error) { status(`启动失败：${error.message}`, "error"); }
})();
