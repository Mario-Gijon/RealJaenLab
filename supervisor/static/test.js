(() => {
  const canvas = document.querySelector("#test-canvas");
  const context = canvas.getContext("2d");
  const empty = document.querySelector("#test-empty");
  const miniPitch = document.querySelector("#test-mini-pitch");
  const lineName = document.querySelector("#test-line-name");
  const pointProgress = document.querySelector("#test-point-progress");
  const familyStatus = document.querySelector("#test-family-status");
  const frameInfo = document.querySelector("#test-frame-info");
  const result = document.querySelector("#test-result");
  const resultSection = document.querySelector("#test-result-section");
  const projectionToggle = document.querySelector("#test-projection-toggle");
  const annotations = new Map();
  const history = [];
  let geometry = null;
  let selectedName = null;
  let draft = null;
  let frame = null;
  let projected = null;
  let projectionVisible = true;

  const api = async (url, options) => {
    const response = await fetch(url, options);
    const body = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(body.detail || "Error de servidor.");
    return body;
  };
  const selectedLine = () => geometry?.lines.find((line) => line.name === selectedName);
  const cloneAnnotation = (annotation) => annotation && ({ line_name: annotation.line_name, points: annotation.points.map((point) => [...point]) });

  function invalidateSolution() {
    projected = null;
    projectionVisible = true;
    projectionToggle.hidden = true;
    resultSection.hidden = true;
    result.textContent = "";
  }
  function applyAnnotation(name, annotation) {
    if (annotation) annotations.set(name, cloneAnnotation(annotation)); else annotations.delete(name);
  }
  function selectLine(name) {
    if (!geometry?.lines.some((line) => line.name === name)) return;
    draft = null;
    selectedName = name;
    updateUI();
    draw();
  }
  function updateUI() {
    const line = selectedLine();
    const replacing = draft?.name === selectedName;
    const completed = selectedName && annotations.has(selectedName) && !replacing;
    lineName.textContent = line?.label || "Selecciona una línea en el campo.";
    lineName.classList.toggle("is-completed", Boolean(completed));
    [...pointProgress.children].forEach((item, index) => {
      const done = Boolean(completed) || (replacing && draft.points.length > index);
      const current = !completed && draft?.name === selectedName && draft.points.length === index;
      item.classList.toggle("is-done", done);
      item.classList.toggle("is-current", current);
    });
    const counts = { x_constant: 0, y_constant: 0 };
    annotations.forEach((_, name) => { const lineItem = geometry?.lines.find((lineItem) => lineItem.name === name); if (lineItem) counts[lineItem.family] += 1; });
    familyStatus.innerHTML = `<span>Longitudinales <strong>${counts.x_constant} / 2</strong></span><span>Transversales <strong>${counts.y_constant} / 2</strong></span>`;
    miniPitch.querySelectorAll("[data-line-name]").forEach((node) => {
      const name = node.dataset.lineName;
      node.classList.toggle("is-selected", name === selectedName);
      node.classList.toggle("is-annotated", annotations.has(name));
    });
    document.querySelector("#test-undo").disabled = !history.length && !draft?.points.length;
    document.querySelector("#test-remove").disabled = !selectedName || !annotations.has(selectedName);
  }
  function drawPolyline(points, closed) {
    if (!points?.length) return;
    context.beginPath(); context.moveTo(points[0][0], points[0][1]);
    points.slice(1).forEach(([x, y]) => context.lineTo(x, y));
    if (closed) context.closePath();
    context.stroke();
  }
  function drawAnnotation(annotation, emphasis = false) {
    context.lineWidth = Math.max(2, canvas.width / 900) * (emphasis ? 1.45 : 1);
    context.beginPath(); context.moveTo(...annotation.points[0]); context.lineTo(...annotation.points[1]); context.stroke();
    annotation.points.forEach(([x, y]) => { context.beginPath(); context.arc(x, y, context.lineWidth * 1.75, 0, Math.PI * 2); context.fill(); });
  }
  function draw() {
    if (!frame) return;
    context.clearRect(0, 0, canvas.width, canvas.height);
    context.drawImage(frame.bitmap, 0, 0);
    context.lineJoin = "round"; context.lineCap = "round";
    if (projected && projectionVisible) {
      context.lineWidth = Math.max(1.5, canvas.width / 1150); context.strokeStyle = "rgba(103, 216, 255, .92)";
      projected.lines.forEach((line) => drawPolyline(line.points, false));
      projected.curves.forEach((curve) => drawPolyline(curve.points, curve.closed));
      context.fillStyle = "rgba(103, 216, 255, .92)";
      projected.points.forEach((point) => { context.beginPath(); context.arc(point.point[0], point.point[1], context.lineWidth * 1.2, 0, Math.PI * 2); context.fill(); });
    }
    context.strokeStyle = "#f2c94c"; context.fillStyle = "#f2c94c";
    annotations.forEach((annotation, name) => drawAnnotation(annotation, name === selectedName));
    if (draft?.points.length) {
      const [x, y] = draft.points[0]; context.lineWidth = Math.max(2, canvas.width / 900) * 1.45;
      context.beginPath(); context.arc(x, y, context.lineWidth * 1.75, 0, Math.PI * 2); context.fill();
    }
  }
  function pathData(points, closed) {
    if (!points?.length) return "";
    return `M ${points.map((point) => point.join(" ")).join(" L ")}${closed ? " Z" : ""}`;
  }
  function renderMiniPitch() {
    if (!geometry) return;
    const { length, width } = geometry.dimensions;
    const lineSvg = geometry.lines.map((line) => `<line class="mini-pitch-line" data-line-name="${line.name}" x1="${line.start[0]}" y1="${line.start[1]}" x2="${line.end[0]}" y2="${line.end[1]}" tabindex="0" role="button" aria-label="${line.label}"></line>`).join("");
    const curveSvg = geometry.curves.map((curve) => `<path class="mini-pitch-mark" d="${pathData(curve.points, curve.closed)}"></path>`).join("");
    const pointSvg = geometry.points.map((point) => `<circle class="mini-pitch-point" cx="${point.point[0]}" cy="${point.point[1]}" r=".7"></circle>`).join("");
    miniPitch.innerHTML = `<svg viewBox="-3 -3 ${length + 6} ${width + 6}" preserveAspectRatio="xMidYMid meet" aria-label="Campo de fútbol interactivo"><rect class="mini-pitch-grass" x="0" y="0" width="${length}" height="${width}"></rect>${curveSvg}${pointSvg}${lineSvg}</svg>`;
    miniPitch.querySelectorAll("[data-line-name]").forEach((node) => {
      node.addEventListener("click", () => selectLine(node.dataset.lineName));
      node.addEventListener("keydown", (event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); selectLine(node.dataset.lineName); } });
    });
  }
  function setEmptyState(message) {
    frame = null; canvas.width = 0; canvas.height = 0; empty.querySelector("span").textContent = message; empty.hidden = false;
    frameInfo.textContent = "Sin fotograma capturado."; draft = null; annotations.clear(); history.length = 0; invalidateSolution(); updateUI();
  }
  function capture() {
    const video = window.SupervisorWorkspace?.getVideo();
    const selected = window.SupervisorWorkspace?.getSelectedVideo();
    if (!selected) return setEmptyState("Abre un vídeo desde Biblioteca para comenzar.");
    if (!video?.videoWidth || video.readyState < 2) return setEmptyState("El vídeo todavía no está listo para capturar un fotograma.");
    video.pause();
    canvas.width = video.videoWidth; canvas.height = video.videoHeight;
    const bitmap = document.createElement("canvas"); bitmap.width = canvas.width; bitmap.height = canvas.height;
    bitmap.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
    frame = { bitmap }; annotations.clear(); history.length = 0; draft = null; invalidateSolution(); empty.hidden = true;
    const frameIndex = selected.fps ? ` · fotograma ${Math.floor(video.currentTime * selected.fps)}` : "";
    frameInfo.textContent = `${selected.displayName || selected.name} · ${video.currentTime.toFixed(3)} s${frameIndex} · ${canvas.width} × ${canvas.height}px`;
    updateUI(); draw();
  }
  async function loadGeometry() {
    try { geometry = await api("/api/field/pitch-lines"); renderMiniPitch(); updateUI(); }
    catch (error) { resultSection.hidden = false; result.textContent = error.message; }
  }
  canvas.addEventListener("click", (event) => {
    if (!frame || !selectedName) return;
    const rect = canvas.getBoundingClientRect();
    const point = [(event.clientX - rect.left) * canvas.width / rect.width, (event.clientY - rect.top) * canvas.height / rect.height];
    if (!draft) draft = { name: selectedName, points: [] };
    draft.points.push(point);
    if (draft.points.length === 2) {
      const before = cloneAnnotation(annotations.get(selectedName));
      const after = { line_name: selectedName, points: draft.points };
      applyAnnotation(selectedName, after);
      history.push({ name: selectedName, before, after: cloneAnnotation(after) });
      draft = null; invalidateSolution();
    }
    updateUI(); draw();
  });
  document.querySelector("#test-undo").addEventListener("click", () => {
    if (draft?.points.length) { draft = null; updateUI(); draw(); return; }
    const operation = history.pop(); if (!operation) return;
    applyAnnotation(operation.name, operation.before); invalidateSolution(); updateUI(); draw();
  });
  document.querySelector("#test-remove").addEventListener("click", () => {
    if (!selectedName || !annotations.has(selectedName)) return;
    const before = cloneAnnotation(annotations.get(selectedName));
    applyAnnotation(selectedName, null); history.push({ name: selectedName, before, after: null }); draft = null; invalidateSolution(); updateUI(); draw();
  });
  document.querySelector("#test-clear").addEventListener("click", () => {
    annotations.clear(); history.length = 0; draft = null; invalidateSolution(); updateUI(); draw();
  });
  document.querySelector("#test-recapture").addEventListener("click", capture);
  projectionToggle.addEventListener("click", () => { projectionVisible = !projectionVisible; projectionToggle.textContent = projectionVisible ? "Ocultar proyección" : "Mostrar proyección"; draw(); });
  document.querySelector("#test-solve").addEventListener("click", async () => {
    const video = window.SupervisorWorkspace?.getVideo(); const selected = window.SupervisorWorkspace?.getSelectedVideo();
    if (!frame || !video || !selected) { resultSection.hidden = false; result.textContent = "Captura un fotograma de un vídeo cargado antes de resolver."; return; }
    try {
      resultSection.hidden = false; result.textContent = "Resolviendo…";
      const payload = { asset_id: selected.assetId, timestamp_seconds: video.currentTime, frame_index: selected.fps ? Math.floor(video.currentTime * selected.fps) : null, video_width: canvas.width, video_height: canvas.height, pitch_length: Number(document.querySelector("#test-pitch-length").value), pitch_width: Number(document.querySelector("#test-pitch-width").value), annotations: [...annotations.values()] };
      const solved = await api("/api/field/calibration/solve", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
      projected = solved.projected_geometry; projectionVisible = true; projectionToggle.hidden = false;
      result.innerHTML = `<strong>Resuelto</strong><dl><div><dt>Anotaciones</dt><dd>${solved.annotation_count}</dd></div><div><dt>Correspondencias</dt><dd>${solved.derived_correspondence_count}</dd></div><div><dt>Error mediano</dt><dd>${solved.metrics.median_px.toFixed(2)} px</dd></div><div><dt>Error p95</dt><dd>${solved.metrics.p95_px.toFixed(2)} px</dd></div><div><dt>Error máximo</dt><dd>${solved.metrics.max_px.toFixed(2)} px</dd></div></dl>`;
      draw();
    } catch (error) { resultSection.hidden = false; result.textContent = `No se pudo resolver: ${error.message}`; }
  });
  window.addEventListener("resize", draw);
  window.SupervisorTest = { enter: capture };
  void loadGeometry();
})();
