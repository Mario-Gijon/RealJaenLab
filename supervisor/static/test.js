(() => {
  const canvas = document.querySelector("#test-canvas");
  const context = canvas.getContext("2d");
  const testVideo = document.querySelector("#test-video");
  const empty = document.querySelector("#test-empty");
  const miniPitch = document.querySelector("#test-mini-pitch");
  const lineModeButton = document.querySelector("#test-line-mode");
  const pointModeButton = document.querySelector("#test-point-mode");
  const annotationGuidance = document.querySelector("#test-annotation-guidance");
  const selectionHeading = document.querySelector("#test-selection-heading");
  const selectionName = document.querySelector("#test-line-name");
  const pointProgress = document.querySelector("#test-point-progress");
  const familyStatus = document.querySelector("#test-family-status");
  const frameInfo = document.querySelector("#test-frame-info");
  const result = document.querySelector("#test-result");
  const resultSection = document.querySelector("#test-result-section");
  const projectionToggle = document.querySelector("#test-projection-toggle");
  const previousButton = document.querySelector("#test-previous-frame");
  const playButton = document.querySelector("#test-play-pause");
  const nextButton = document.querySelector("#test-next-frame");
  const timeline = document.querySelector("#test-timeline");
  const currentTimeLabel = document.querySelector("#test-current-time");
  const totalTimeLabel = document.querySelector("#test-total-time");
  const frameInput = document.querySelector("#test-frame-input");
  const frameTotal = document.querySelector("#test-frame-total");
  const lockButton = document.querySelector("#test-lock-frame");
  const lineAnnotations = new Map();
  const pointAnnotations = new Map();
  const history = [];
  let geometry = null;
  let mode = "line";
  let selectedName = null;
  let draft = null;
  let frameLocked = false;
  let fixedTimestamp = null;
  let fixedFrame = null;
  let projected = null;
  let projectionVisible = true;
  let pendingStartTime = null;

  const api = async (url, options) => {
    const response = await fetch(url, options);
    const body = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(body.detail || "Error de servidor.");
    return body;
  };
  const selectedItem = () => geometry?.[mode === "line" ? "lines" : "points"].find((item) => item.name === selectedName);
  const annotationMap = (kind) => kind === "line" ? lineAnnotations : pointAnnotations;
  const cloneAnnotation = (kind, annotation) => annotation && (kind === "line"
    ? { line_name: annotation.line_name, points: annotation.points.map((point) => [...point]) }
    : { point_name: annotation.point_name, point: [...annotation.point] });
  const formatTime = (seconds) => [Math.floor(Math.max(0, seconds || 0) / 3600), Math.floor((Math.max(0, seconds || 0) % 3600) / 60), Math.floor(Math.max(0, seconds || 0) % 60)].map((part) => String(part).padStart(2, "0")).join(":");

  function resetSolution() {
    projected = null; projectionVisible = true; projectionToggle.hidden = true; resultSection.hidden = true; result.textContent = "";
  }
  function resetAnnotations() {
    lineAnnotations.clear(); pointAnnotations.clear(); history.length = 0; draft = null; resetSolution();
  }
  function applyAnnotation(kind, name, annotation) {
    const target = annotationMap(kind);
    if (annotation) target.set(name, cloneAnnotation(kind, annotation)); else target.delete(name);
  }
  function selectItem(name) {
    if (!frameLocked) return;
    const collection = geometry?.[mode === "line" ? "lines" : "points"] || [];
    if (!collection.some((item) => item.name === name)) return;
    draft = null; selectedName = name; updateUI(); draw();
  }
  function setMode(nextMode) {
    if (!frameLocked) return;
    mode = nextMode; selectedName = null; draft = null; updateUI(); draw();
  }
  function setEmpty(message) {
    empty.querySelector("span").textContent = message;
    empty.hidden = false;
  }
  function updatePlayback() {
    const selected = window.SupervisorWorkspace?.getSelectedVideo();
    const duration = Number.isFinite(testVideo.duration) ? testVideo.duration : (selected?.duration || 0);
    const current = testVideo.currentTime || 0;
    timeline.max = duration; timeline.value = Math.min(current, duration);
    currentTimeLabel.textContent = formatTime(current); totalTimeLabel.textContent = formatTime(duration);
    if (selected?.fps) {
      const maximum = selected.frame_count ? selected.frame_count - 1 : Math.floor(duration * selected.fps);
      const frame = Math.max(0, Math.min(Math.floor(current * selected.fps), maximum));
      frameInput.value = String(frame); frameInput.max = Number.isFinite(maximum) ? String(maximum) : ""; frameTotal.textContent = selected.frame_count ? String(selected.frame_count) : "—";
    } else { frameInput.value = ""; frameInput.max = ""; frameTotal.textContent = "—"; }
  }
  function updateControls() {
    const selected = window.SupervisorWorkspace?.getSelectedVideo();
    const ready = Boolean(selected && testVideo.readyState >= 2);
    const fpsReady = ready && Boolean(selected?.fps);
    previousButton.disabled = !fpsReady; nextButton.disabled = !fpsReady; frameInput.disabled = !fpsReady;
    playButton.disabled = !ready; timeline.disabled = !ready; lockButton.disabled = !ready;
    playButton.textContent = testVideo.paused ? "▶" : "❚❚";
    playButton.setAttribute("aria-label", testVideo.paused ? "Reproducir" : "Pausar");
    lockButton.textContent = frameLocked ? "Cambiar fotograma" : "Fijar fotograma";
    canvas.classList.toggle("is-active", frameLocked);
  }
  function updateUI() {
    const item = selectedItem();
    const selectedAnnotations = annotationMap(mode);
    const replacing = mode === "line" && draft?.name === selectedName;
    const complete = Boolean(selectedName && selectedAnnotations.has(selectedName) && !replacing);
    selectionHeading.textContent = mode === "line" ? "Línea seleccionada" : "Punto seleccionado";
    selectionName.textContent = item?.label || `Selecciona ${mode === "line" ? "una línea" : "un punto"} en el campo.`;
    selectionName.classList.toggle("is-completed", complete);
    pointProgress.classList.toggle("is-point-mode", mode === "point");
    pointProgress.children[0].textContent = mode === "line" ? "Primer punto" : "Punto";
    pointProgress.children[1].textContent = "Segundo punto";
    [...pointProgress.children].forEach((node, index) => {
      const done = complete || (replacing && draft.points.length > index);
      const current = !complete && mode === "line" && draft?.name === selectedName && draft.points.length === index;
      node.classList.toggle("is-done", done); node.classList.toggle("is-current", current);
    });
    if (mode === "point") pointProgress.children[0].classList.toggle("is-done", complete);
    if (!frameLocked) annotationGuidance.textContent = "Fija un fotograma para comenzar a anotar.";
    else if (!item) annotationGuidance.textContent = `Selecciona ${mode === "line" ? "una línea" : "un punto"} en el mini campo.`;
    else if (mode === "point") annotationGuidance.textContent = complete ? "Punto completado. Puedes seleccionar otro." : "Haz clic en su posición en el fotograma.";
    else if (complete) annotationGuidance.textContent = "Línea completada. Puedes seleccionar otra.";
    else if (draft?.points.length === 1) annotationGuidance.textContent = "Marca el segundo punto sobre esta línea.";
    else annotationGuidance.textContent = "Marca dos puntos sobre esta línea.";
    const counts = { x_constant: 0, y_constant: 0 };
    lineAnnotations.forEach((_, name) => { const line = geometry?.lines.find((candidate) => candidate.name === name); if (line) counts[line.family] += 1; });
    familyStatus.innerHTML = `<span>Líneas <strong>${lineAnnotations.size}</strong></span><span>Puntos <strong>${pointAnnotations.size}</strong></span><span>Longitudinales <strong>${counts.x_constant}</strong></span><span>Transversales <strong>${counts.y_constant}</strong></span>`;
    lineModeButton.classList.toggle("is-active", mode === "line"); pointModeButton.classList.toggle("is-active", mode === "point");
    lineModeButton.disabled = !frameLocked; pointModeButton.disabled = !frameLocked;
    miniPitch.classList.toggle("is-point-mode", mode === "point"); miniPitch.classList.toggle("is-frame-unfixed", !frameLocked);
    miniPitch.querySelectorAll("[data-line-name]").forEach((node) => { const name = node.dataset.lineName; node.classList.toggle("is-selected", mode === "line" && name === selectedName); node.classList.toggle("is-annotated", lineAnnotations.has(name)); });
    miniPitch.querySelectorAll("[data-point-name]").forEach((node) => { const name = node.dataset.pointName; node.classList.toggle("is-selected", mode === "point" && name === selectedName); node.classList.toggle("is-annotated", pointAnnotations.has(name)); });
    document.querySelector("#test-undo").disabled = !history.length && !draft?.points.length;
    document.querySelector("#test-remove").disabled = !selectedName || !selectedAnnotations.has(selectedName);
    document.querySelector("#test-clear").disabled = !frameLocked;
    document.querySelector("#test-remove").textContent = mode === "line" ? "Eliminar línea" : "Eliminar punto";
    document.querySelector("#test-solve").disabled = !frameLocked;
    updateControls();
  }
  function drawPolyline(points, closed) {
    if (!points?.length) return;
    context.beginPath(); context.moveTo(points[0][0], points[0][1]); points.slice(1).forEach(([x, y]) => context.lineTo(x, y)); if (closed) context.closePath(); context.stroke();
  }
  function draw() {
    if (!frameLocked || !canvas.width || !canvas.height) return;
    context.clearRect(0, 0, canvas.width, canvas.height); context.lineJoin = "round"; context.lineCap = "round";
    if (projected && projectionVisible) {
      context.lineWidth = Math.max(1.5, canvas.width / 1150); context.strokeStyle = "rgba(103, 216, 255, .92)";
      projected.lines.forEach((line) => drawPolyline(line.points, false)); projected.curves.forEach((curve) => drawPolyline(curve.points, curve.closed));
      context.fillStyle = "rgba(103, 216, 255, .92)"; projected.points.forEach((point) => { context.beginPath(); context.arc(point.point[0], point.point[1], context.lineWidth * 1.2, 0, Math.PI * 2); context.fill(); });
    }
    const scale = Math.max(2, canvas.width / 900); context.strokeStyle = "#f2c94c"; context.fillStyle = "#f2c94c";
    lineAnnotations.forEach((annotation, name) => { context.lineWidth = scale * (mode === "line" && name === selectedName ? 1.45 : 1); context.beginPath(); context.moveTo(...annotation.points[0]); context.lineTo(...annotation.points[1]); context.stroke(); annotation.points.forEach(([x, y]) => { context.beginPath(); context.arc(x, y, context.lineWidth * 1.7, 0, Math.PI * 2); context.fill(); }); });
    pointAnnotations.forEach((annotation, name) => { const [x, y] = annotation.point; context.beginPath(); context.arc(x, y, scale * (mode === "point" && name === selectedName ? 2.4 : 1.9), 0, Math.PI * 2); context.fill(); });
    if (draft?.points.length) { const [x, y] = draft.points[0]; context.beginPath(); context.arc(x, y, scale * 2.4, 0, Math.PI * 2); context.fill(); }
  }
  function pathData(points, closed) { return points?.length ? `M ${points.map((point) => point.join(" ")).join(" L ")}${closed ? " Z" : ""}` : ""; }
  function renderMiniPitch() {
    if (!geometry) return;
    const { length, width } = geometry.dimensions;
    const lines = geometry.lines.map((line) => `<line class="mini-pitch-line" data-line-name="${line.name}" x1="${line.start[0]}" y1="${line.start[1]}" x2="${line.end[0]}" y2="${line.end[1]}" tabindex="0" role="button" aria-label="${line.label}"></line>`).join("");
    const curves = geometry.curves.map((curve) => `<path class="mini-pitch-mark" d="${pathData(curve.points, curve.closed)}"></path>`).join("");
    const points = geometry.points.map((point) => `<g class="mini-pitch-point-hit" data-point-name="${point.name}" tabindex="0" role="button" aria-label="${point.label}"><circle class="mini-pitch-point-target" cx="${point.point[0]}" cy="${point.point[1]}" r="3"></circle><circle class="mini-pitch-point" cx="${point.point[0]}" cy="${point.point[1]}" r=".9"></circle></g>`).join("");
    miniPitch.innerHTML = `<svg viewBox="-3 -3 ${length + 6} ${width + 6}" preserveAspectRatio="xMidYMid meet" aria-label="Campo de fútbol interactivo"><rect class="mini-pitch-grass" x="0" y="0" width="${length}" height="${width}"></rect>${curves}${lines}${points}</svg>`;
    miniPitch.querySelectorAll("[data-line-name], [data-point-name]").forEach((node) => { const kind = node.dataset.lineName ? "line" : "point"; const name = node.dataset.lineName || node.dataset.pointName; node.addEventListener("click", () => { if (frameLocked && mode === kind) selectItem(name); }); node.addEventListener("keydown", (event) => { if ((event.key === "Enter" || event.key === " ") && frameLocked && mode === kind) { event.preventDefault(); selectItem(name); } }); });
  }
  function clearOverlay() { context.clearRect(0, 0, canvas.width, canvas.height); }
  function unlockFrame() {
    if (!frameLocked) return;
    frameLocked = false; fixedTimestamp = null; fixedFrame = null; selectedName = null; resetAnnotations(); clearOverlay(); frameInfo.textContent = "Navega por el vídeo y fija un fotograma para anotar."; updateUI();
  }
  function lockFrame() {
    const selected = window.SupervisorWorkspace?.getSelectedVideo();
    if (!selected || testVideo.readyState < 2) return;
    testVideo.pause(); frameLocked = true; fixedTimestamp = testVideo.currentTime; fixedFrame = selected.fps ? Math.floor(fixedTimestamp * selected.fps) : null;
    canvas.width = testVideo.videoWidth; canvas.height = testVideo.videoHeight; clearOverlay(); empty.hidden = true;
    frameInfo.textContent = `${selected.displayName || selected.name} · ${fixedTimestamp.toFixed(3)} s${fixedFrame !== null ? ` · fotograma ${fixedFrame}` : ""} · ${canvas.width} × ${canvas.height}px`;
    updateUI(); draw();
  }
  function stepFrame(direction) {
    const selected = window.SupervisorWorkspace?.getSelectedVideo(); if (!selected?.fps) return;
    unlockFrame(); testVideo.pause(); const duration = Number.isFinite(testVideo.duration) ? testVideo.duration : Infinity; testVideo.currentTime = Math.max(0, Math.min(duration, testVideo.currentTime + direction / selected.fps));
  }
  function seekFrame(value) {
    const selected = window.SupervisorWorkspace?.getSelectedVideo(); if (!selected?.fps || !Number.isFinite(value)) return;
    unlockFrame(); const maximum = selected.frame_count ? selected.frame_count - 1 : Math.floor((testVideo.duration || 0) * selected.fps); const frame = Math.max(0, Math.min(Math.floor(value), maximum)); testVideo.pause(); testVideo.currentTime = frame / selected.fps;
  }
  function initialiseVideo() {
    const selected = window.SupervisorWorkspace?.getSelectedVideo(); const sourceVideo = window.SupervisorWorkspace?.getVideo();
    if (!selected) { testVideo.removeAttribute("src"); testVideo.load(); setEmpty("Abre un vídeo desde Biblioteca para comenzar."); return; }
    const sourceChanged = testVideo.dataset.assetId !== selected.assetId;
    pendingStartTime = sourceVideo?.currentTime || 0;
    if (sourceChanged) { unlockFrame(); testVideo.dataset.assetId = selected.assetId; testVideo.src = selected.playbackUrl; testVideo.load(); setEmpty("El vídeo todavía no está listo."); }
    else if (testVideo.readyState >= 2) { if (!frameLocked) testVideo.currentTime = pendingStartTime; empty.hidden = true; updateUI(); }
  }
  canvas.addEventListener("click", (event) => {
    if (!frameLocked || !selectedName) return;
    const rect = canvas.getBoundingClientRect(); const point = [(event.clientX - rect.left) * canvas.width / rect.width, (event.clientY - rect.top) * canvas.height / rect.height];
    if (mode === "point") { const before = cloneAnnotation("point", pointAnnotations.get(selectedName)); const after = { point_name: selectedName, point }; applyAnnotation("point", selectedName, after); history.push({ kind: "point", name: selectedName, before, after: cloneAnnotation("point", after) }); resetSolution(); }
    else { if (!draft) draft = { name: selectedName, points: [] }; draft.points.push(point); if (draft.points.length === 2) { const before = cloneAnnotation("line", lineAnnotations.get(selectedName)); const after = { line_name: selectedName, points: draft.points }; applyAnnotation("line", selectedName, after); history.push({ kind: "line", name: selectedName, before, after: cloneAnnotation("line", after) }); draft = null; resetSolution(); } }
    updateUI(); draw();
  });
  lineModeButton.addEventListener("click", () => setMode("line")); pointModeButton.addEventListener("click", () => setMode("point"));
  document.querySelector("#test-undo").addEventListener("click", () => { if (draft?.points.length) { draft = null; updateUI(); draw(); return; } const operation = history.pop(); if (!operation) return; applyAnnotation(operation.kind, operation.name, operation.before); resetSolution(); updateUI(); draw(); });
  document.querySelector("#test-remove").addEventListener("click", () => { const target = annotationMap(mode); if (!selectedName || !target.has(selectedName)) return; const before = cloneAnnotation(mode, target.get(selectedName)); applyAnnotation(mode, selectedName, null); history.push({ kind: mode, name: selectedName, before, after: null }); draft = null; resetSolution(); updateUI(); draw(); });
  document.querySelector("#test-clear").addEventListener("click", () => { resetAnnotations(); updateUI(); draw(); });
  document.querySelector("#test-solve").addEventListener("click", async () => {
    const selected = window.SupervisorWorkspace?.getSelectedVideo();
    if (!frameLocked || !selected) return;
    try {
      resultSection.hidden = false; result.textContent = "Resolviendo…";
      const payload = {
        asset_id: selected.assetId,
        timestamp_seconds: fixedTimestamp,
        frame_index: fixedFrame,
        video_width: canvas.width,
        video_height: canvas.height,
        pitch_length: Number(document.querySelector("#test-pitch-length").value),
        pitch_width: Number(document.querySelector("#test-pitch-width").value),
        annotations: [...lineAnnotations.values()],
        point_annotations: [...pointAnnotations.values()],
      };
      const solved = await api("/api/field/calibration/solve", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
      projected = solved.projected_geometry; projectionVisible = true; projectionToggle.hidden = false;
      result.innerHTML = `<strong>Resuelto</strong><dl><div><dt>Líneas</dt><dd>${solved.line_annotation_count}</dd></div><div><dt>Puntos</dt><dd>${solved.point_annotation_count}</dd></div><div><dt>Intersecciones de líneas</dt><dd>${solved.derived_correspondence_count}</dd></div><div><dt>Correspondencias directas</dt><dd>${solved.direct_correspondence_count}</dd></div><div><dt>Total correspondencias</dt><dd>${solved.correspondence_count}</dd></div><div><dt>Error mediano</dt><dd>${solved.metrics.median_px.toFixed(2)} px</dd></div><div><dt>Error p95</dt><dd>${solved.metrics.p95_px.toFixed(2)} px</dd></div><div><dt>Error máximo</dt><dd>${solved.metrics.max_px.toFixed(2)} px</dd></div></dl>`;
      draw();
    } catch (error) { resultSection.hidden = false; result.textContent = `No se pudo resolver: ${error.message}`; }
  });
  projectionToggle.addEventListener("click", () => { projectionVisible = !projectionVisible; projectionToggle.textContent = projectionVisible ? "Ocultar proyección" : "Mostrar proyección"; draw(); });
  previousButton.addEventListener("click", () => stepFrame(-1)); nextButton.addEventListener("click", () => stepFrame(1));
  playButton.addEventListener("click", () => { if (testVideo.paused) void testVideo.play(); else testVideo.pause(); });
  timeline.addEventListener("input", () => { unlockFrame(); testVideo.currentTime = Number(timeline.value); });
  frameInput.addEventListener("keydown", (event) => { if (event.key === "Enter") { event.preventDefault(); seekFrame(Number(frameInput.value)); } }); frameInput.addEventListener("change", () => seekFrame(Number(frameInput.value)));
  lockButton.addEventListener("click", () => { if (frameLocked) unlockFrame(); else lockFrame(); });
  testVideo.addEventListener("loadeddata", () => { if (pendingStartTime !== null) { testVideo.currentTime = pendingStartTime; pendingStartTime = null; } empty.hidden = true; updatePlayback(); updateUI(); });
  testVideo.addEventListener("timeupdate", updatePlayback); testVideo.addEventListener("seeked", updatePlayback);
  testVideo.addEventListener("play", () => { unlockFrame(); updateUI(); }); testVideo.addEventListener("pause", updateUI);
  testVideo.addEventListener("error", () => { setEmpty("No se pudo cargar el vídeo seleccionado."); updateUI(); });
  window.addEventListener("resize", draw);
  window.SupervisorTest = { enter: initialiseVideo, leave: () => { const sourceVideo = window.SupervisorWorkspace?.getVideo(); if (sourceVideo && testVideo.readyState >= 1) sourceVideo.currentTime = testVideo.currentTime; } };
  api("/api/field/pitch-lines").then((data) => { geometry = data; renderMiniPitch(); updateUI(); }).catch((error) => { resultSection.hidden = false; result.textContent = error.message; });
})();
