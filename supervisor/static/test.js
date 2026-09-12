(() => {
  const canvas = document.querySelector("#test-canvas");
  const context = canvas.getContext("2d");
  const empty = document.querySelector("#test-empty");
  const select = document.querySelector("#test-line-select");
  const status = document.querySelector("#test-status");
  const familyStatus = document.querySelector("#test-family-status");
  const frameInfo = document.querySelector("#test-frame-info");
  const result = document.querySelector("#test-result");
  const annotations = new Map();
  let lines = [];
  let frame = null;
  let projected = null;

  const api = async (url, options) => {
    const response = await fetch(url, options);
    const body = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(body.detail || "Error de servidor.");
    return body;
  };

  function selectedLine() { return lines.find((line) => line.name === select.value); }
  function updateStatus() {
    const current = annotations.get(select.value);
    status.textContent = current
      ? `${selectedLine()?.label || select.value}: 2 puntos · selecciona otra línea o vuelve a clicar para reemplazar.`
      : `${selectedLine()?.label || "Línea"}: ${frame?.pending?.length || 0} / 2 puntos.`;
    const counts = { x_constant: 0, y_constant: 0 };
    annotations.forEach((_, name) => { const line = lines.find((item) => item.name === name); if (line) counts[line.family] += 1; });
    familyStatus.textContent = `x_constant: ${counts.x_constant} / 2 · y_constant: ${counts.y_constant} / 2`;
  }
  function draw() {
    if (!frame) return;
    context.clearRect(0, 0, canvas.width, canvas.height);
    context.drawImage(frame.bitmap, 0, 0);
    context.lineWidth = Math.max(2, canvas.width / 900);
    if (projected) {
      context.strokeStyle = "#67d8ff";
      projected.lines.forEach((line) => drawPolyline(line.points, false));
      projected.curves.forEach((curve) => drawPolyline(curve.points, curve.closed));
      context.fillStyle = "#67d8ff";
      projected.points.forEach((point) => { context.beginPath(); context.arc(point.point[0], point.point[1], context.lineWidth * 1.4, 0, Math.PI * 2); context.fill(); });
    }
    context.strokeStyle = "#f2c94c";
    context.fillStyle = "#f2c94c";
    annotations.forEach((annotation) => {
      context.beginPath(); context.moveTo(annotation.points[0][0], annotation.points[0][1]); context.lineTo(annotation.points[1][0], annotation.points[1][1]); context.stroke();
      annotation.points.forEach(([x, y]) => { context.beginPath(); context.arc(x, y, context.lineWidth * 2, 0, Math.PI * 2); context.fill(); });
    });
    if (frame.pending?.length) {
      context.fillStyle = "#f2c94c"; context.beginPath(); context.arc(frame.pending[0][0], frame.pending[0][1], context.lineWidth * 2, 0, Math.PI * 2); context.fill();
    }
  }
  function drawPolyline(points, closed) {
    if (!points?.length) return;
    context.beginPath(); context.moveTo(points[0][0], points[0][1]);
    points.slice(1).forEach(([x, y]) => context.lineTo(x, y));
    if (closed) context.closePath();
    context.stroke();
  }
  function capture() {
    const video = window.SupervisorWorkspace?.getVideo();
    const selected = window.SupervisorWorkspace?.getSelectedVideo();
    if (!video || !selected || !video.videoWidth || video.readyState < 2) {
      frame = null; empty.hidden = false; frameInfo.textContent = "Sin vídeo cargado. Abre un vídeo desde la biblioteca."; draw(); return;
    }
    video.pause();
    canvas.width = video.videoWidth; canvas.height = video.videoHeight;
    const bitmap = document.createElement("canvas"); bitmap.width = canvas.width; bitmap.height = canvas.height;
    bitmap.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
    frame = { bitmap, pending: [] }; projected = null; annotations.clear(); empty.hidden = true;
    const fps = selected.fps ? ` · frame ${Math.floor(video.currentTime * selected.fps)}` : "";
    frameInfo.textContent = `${selected.displayName || selected.name} · ${video.currentTime.toFixed(3)} s${fps} · ${canvas.width} × ${canvas.height}px`;
    result.textContent = ""; updateStatus(); draw();
  }
  async function loadLines() {
    try { const data = await api("/api/field/pitch-lines"); lines = data.lines; select.innerHTML = lines.map((line) => `<option value="${line.name}">${line.label} · ${line.family}</option>`).join(""); updateStatus(); }
    catch (error) { result.textContent = error.message; }
  }
  canvas.addEventListener("click", (event) => {
    if (!frame) return;
    const rect = canvas.getBoundingClientRect();
    const point = [(event.clientX - rect.left) * canvas.width / rect.width, (event.clientY - rect.top) * canvas.height / rect.height];
    const name = select.value;
    if (annotations.has(name)) annotations.delete(name);
    frame.pending.push(point);
    if (frame.pending.length === 2) { annotations.set(name, { line_name: name, points: frame.pending }); frame.pending = []; }
    updateStatus(); draw();
  });
  select.addEventListener("change", () => { if (frame) frame.pending = []; updateStatus(); draw(); });
  document.querySelector("#test-undo").addEventListener("click", () => { if (frame?.pending?.length) frame.pending.pop(); else annotations.delete(select.value); updateStatus(); draw(); });
  document.querySelector("#test-remove").addEventListener("click", () => { annotations.delete(select.value); if (frame) frame.pending = []; updateStatus(); draw(); });
  document.querySelector("#test-clear").addEventListener("click", () => { annotations.clear(); if (frame) frame.pending = []; projected = null; result.textContent = ""; updateStatus(); draw(); });
  document.querySelector("#test-recapture").addEventListener("click", () => { window.SupervisorWorkspace?.setWorkspace("video"); });
  document.querySelector("#test-solve").addEventListener("click", async () => {
    const video = window.SupervisorWorkspace?.getVideo(); const selected = window.SupervisorWorkspace?.getSelectedVideo();
    if (!frame || !video || !selected) return (result.textContent = "Captura primero un frame de un vídeo cargado.");
    try {
      result.textContent = "Resolviendo…";
      const payload = { asset_id: selected.assetId, timestamp_seconds: video.currentTime, frame_index: selected.fps ? Math.floor(video.currentTime * selected.fps) : null, video_width: canvas.width, video_height: canvas.height, pitch_length: Number(document.querySelector("#test-pitch-length").value), pitch_width: Number(document.querySelector("#test-pitch-width").value), annotations: [...annotations.values()] };
      const solved = await api("/api/field/calibration/solve", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
      projected = solved.projected_geometry; result.innerHTML = `<strong>Resuelto</strong><br>${solved.annotation_count} anotaciones · ${solved.derived_correspondence_count} correspondencias<br>mediana ${solved.metrics.median_px.toFixed(2)} px · p95 ${solved.metrics.p95_px.toFixed(2)} px · máximo ${solved.metrics.max_px.toFixed(2)} px`; draw();
    } catch (error) { result.textContent = `No se pudo resolver: ${error.message}`; }
  });
  window.addEventListener("resize", draw);
  window.SupervisorTest = { enter: capture };
  void loadLines();
})();
