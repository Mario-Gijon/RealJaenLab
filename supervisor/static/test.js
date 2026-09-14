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
  const cameraModelModes = document.querySelector("#test-camera-model-modes");
  const homographyButton = document.querySelector("#test-model-homography");
  const radialButton = document.querySelector("#test-model-radial");
  const anisotropicButton = document.querySelector("#test-model-anisotropic");
  const cameraBest = document.querySelector("#test-camera-best");
  const pitchLengthInput = document.querySelector("#test-pitch-length");
  const pitchWidthInput = document.querySelector("#test-pitch-width");
  const cameraStateProfile = document.querySelector("#test-camera-state-profile");
  const saveHumanButton = document.querySelector("#test-save-human");
  const analyzeCameraStatesButton = document.querySelector("#test-analyze-camera-states");
  const resetCameraStatesButton = document.querySelector("#test-reset-camera-states");
  const cameraStateList = document.querySelector("#test-camera-state-list");
  const cameraStateAnalysisElement = document.querySelector("#test-camera-state-analysis");
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
  let cameraModel = "pure_homography";
  let cameraModelProbe = null;
  let currentSolve = null;
  let humanStates = [];
  let cameraStateReferenceId = null;
  let cameraStateAnalysis = null;
  let cameraStateFieldProfile = null;
  let cameraStateVideoAssetId = null;
  let diagnosticFocus = null;
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
  const cloneJson = (value) => typeof structuredClone === "function" ? structuredClone(value) : JSON.parse(JSON.stringify(value));
  const cameraStateModelLabel = (name) => ({ translation: "Traslación", similarity: "Similitud", affine: "Afín", homography: "Homografía" }[name] || name || "—");
  const projectionModelLabel = (name) => ({ pure_homography: "Homografía", radial_1p: "Radial 1P", anisotropic_2p: "Anisotrópica 2P" }[name] || name || "—");
  const cameraStateDescription = (name) => ({ translation: "desplazamiento", similarity: "zoom uniforme + rotación + desplazamiento", affine: "escala/ejes distintos + cizalla + desplazamiento", homography: "cambio proyectivo de la cámara" }[name] || "modelo no identificado");
  const numberText = (value, digits = 2) => finite(value) ? Number(value).toFixed(digits) : "—";
  const stateFrameText = (state) => `Fotograma ${state.frame_index ?? "—"} · ${formatTime(state.timestamp_seconds)}${state.state_id === cameraStateReferenceId ? " · referencia" : ""}`;
  const anisotropicGeometryAvailable = () => Boolean(
    currentSolve?.camera_model_probe?.models?.anisotropic_2p?.projected_geometry
  );

  function clearCameraStateAnalysis() {
    cameraStateAnalysis = null;
  }
  function resetCameraStateExperiment() {
    humanStates = [];
    cameraStateReferenceId = null;
    cameraStateAnalysis = null;
    cameraStateFieldProfile = null;
    cameraStateVideoAssetId = null;
    renderCameraStateExperiment();
  }
  function saveCurrentHumanState() {
    if (!frameLocked || !Number.isInteger(fixedFrame) || !anisotropicGeometryAvailable()) return;
    const stateId = `frame_${fixedFrame}`;
    const state = {
      state_id: stateId,
      frame_index: fixedFrame,
      timestamp_seconds: fixedTimestamp,
      projection_model: "anisotropic_2p",
      solve: cloneJson(currentSolve),
    };
    const existing = humanStates.findIndex((candidate) => candidate.state_id === stateId);
    if (existing >= 0) humanStates.splice(existing, 1, state); else humanStates.push(state);
    if (!cameraStateFieldProfile) {
      cameraStateFieldProfile = {
        length: currentSolve.pitch_length,
        width: currentSolve.pitch_width,
      };
    }
    cameraStateVideoAssetId = window.SupervisorWorkspace?.getSelectedVideo()?.assetId || null;
    if (!cameraStateReferenceId) cameraStateReferenceId = stateId;
    clearCameraStateAnalysis();
    renderCameraStateExperiment();
  }
  function setCameraStateReference(stateId) {
    if (!humanStates.some((state) => state.state_id === stateId)) return;
    cameraStateReferenceId = stateId;
    clearCameraStateAnalysis();
    renderCameraStateExperiment();
  }
  function removeHumanState(stateId) {
    humanStates = humanStates.filter((state) => state.state_id !== stateId);
    if (cameraStateReferenceId === stateId) cameraStateReferenceId = humanStates[0]?.state_id || null;
    if (!humanStates.length) {
      cameraStateFieldProfile = null;
      cameraStateVideoAssetId = null;
    }
    clearCameraStateAnalysis();
    renderCameraStateExperiment();
  }
  function transformDetails(transform) {
    if (!transform) return "No se pudo interpretar la transformación.";
    const shift = transform.center_shift_px || [];
    const normalized = transform.center_shift_normalized || [];
    const projective = transform.projective_terms_per_diagonal || [];
    return `Centro Δ ${numberText(shift[0])}, ${numberText(shift[1])} px · normalizado ${numberText(normalized[0], 4)}, ${numberText(normalized[1], 4)} · escala ${numberText(transform.scale_geometric, 4)} · log ${numberText(transform.log_scale, 4)} · mayor/menor ${numberText(transform.scale_major, 4)} / ${numberText(transform.scale_minor, 4)} · anisotropía ${numberText(transform.anisotropy_ratio, 4)} · rotación ${numberText(transform.rotation_deg, 2)}° · proyectivo/diagonal ${numberText(projective[0], 4)}, ${numberText(projective[1], 4)}`;
  }
  function relationMarkup(relation) {
    const modelRows = ["translation", "similarity", "affine", "homography"].map((name) => {
      const model = relation.models?.[name] || {};
      if (model.error) return `<div class="test-camera-state-model"><strong>${cameraStateModelLabel(name)}</strong> · ${model.error}</div>`;
      const summary = model.residual_summary || {};
      const recommended = name === relation.recommended_model;
      return `<div class="test-camera-state-model ${recommended ? "is-recommended" : ""}"><strong>${cameraStateModelLabel(name)}${recommended ? " · recomendado" : ""}</strong><br>Mediana ${metricText(summary.median_px)} · P95 ${metricText(summary.p95_px)}${recommended ? ` · Máximo ${metricText(summary.max_px)}<br>${transformDetails(model.transform)}` : ""}</div>`;
    }).join("");
    const fallbackWarning = relation.correspondence_scope === "all_field_fallback" ? "Comparación con poco solapamiento visible; se usó geometría extrapolada del campo completo." : "";
    return `<article class="test-camera-state-relation"><strong>${relation.source_state_id} (fotograma ${relation.source_frame_index ?? "—"}) → ${relation.target_state_id} (fotograma ${relation.target_frame_index ?? "—"}) · ${cameraStateModelLabel(relation.recommended_model)}</strong><small>${cameraStateDescription(relation.recommended_model)} · ${relation.correspondence_scope} · ${relation.correspondence_count} correspondencias · solapamiento visible ${relation.visible_overlap_count}</small>${fallbackWarning ? `<p class="test-camera-state-warning">${fallbackWarning}</p>` : ""}<div class="test-camera-state-models">${modelRows}</div></article>`;
  }
  function analysisMarkup(analysis) {
    if (!analysis) return "";
    const profile = analysis.pitch_profile || {};
    const reference = (analysis.states || []).find((state) => state.state_id === analysis.reference_state_id);
    const warnings = (analysis.warnings || []).map((warning) => `<p class="test-camera-state-warning">${warning}</p>`).join("");
    const relations = (analysis.relations_to_reference || []).map(relationMarkup).join("") || "<p>La referencia es el único estado disponible.</p>";
    const pairs = (analysis.pairwise_summary || []).map((pair) => `<article class="test-camera-state-pair"><strong>${pair.source_state_id} → ${pair.target_state_id}</strong><small>${cameraStateModelLabel(pair.recommended_model)} · P95 ${metricText(pair.p95_px)} · ${pair.correspondence_scope} · solapamiento visible ${pair.visible_overlap_count}</small></article>`).join("");
    const counts = Object.entries(analysis.recommendation_counts || {}).map(([name, count]) => `${cameraStateModelLabel(name)}: ${count}`).join(" · ");
    return `<h3>Camera State · Análisis</h3><p class="test-diagnostic-caption">Experimento diagnóstico. Buscamos la representación numérica mínima que explique cambios de encuadre/zoom entre calibraciones HUMAN del mismo campo.</p><dl><div><dt>HUMAN</dt><dd>${analysis.state_count ?? 0}</dd></div><div><dt>Referencia</dt><dd>Fotograma ${reference?.frame_index ?? analysis.reference_state_id ?? "—"}</dd></div><div><dt>Proyección</dt><dd>${projectionModelLabel(analysis.projection_model)}</dd></div><div><dt>Campo fijo</dt><dd>${numberText(profile.length, 1)} × ${numberText(profile.width, 1)} m</dd></div><div><dt>Dominante</dt><dd>${cameraStateModelLabel(analysis.dominant_recommended_model)}</dd></div></dl><p class="test-diagnostic-caption">${counts}</p>${warnings}<h3>Relaciones con referencia</h3>${relations}<h3>Resumen por pares</h3>${pairs || "<p>Sin pares para resumir.</p>"}`;
  }
  function renderCameraStateExperiment() {
    const hasStates = humanStates.length > 0;
    pitchLengthInput.disabled = hasStates;
    pitchWidthInput.disabled = hasStates;
    cameraStateProfile.textContent = hasStates
      ? `Perfil FIELD bloqueado: ${numberText(cameraStateFieldProfile?.length, 1)} × ${numberText(cameraStateFieldProfile?.width, 1)} m · ${humanStates.length} HUMAN.`
      : "Sin experimento HUMAN. Las dimensiones del campo siguen editables.";
    saveHumanButton.disabled = !frameLocked || !Number.isInteger(fixedFrame) || !anisotropicGeometryAvailable();
    analyzeCameraStatesButton.disabled = humanStates.length < 2;
    resetCameraStatesButton.disabled = !hasStates;
    cameraStateList.innerHTML = humanStates.map((state) => `<article class="test-camera-state-card ${state.state_id === cameraStateReferenceId ? "is-reference" : ""}"><strong>CAMERA STATE · HUMAN</strong><small>${stateFrameText(state)} · ${projectionModelLabel(state.projection_model)}</small><div><button class="button button-secondary" type="button" data-camera-state-reference="${state.state_id}" ${state.state_id === cameraStateReferenceId ? "disabled" : ""}>Usar como referencia</button><button class="button button-secondary" type="button" data-camera-state-remove="${state.state_id}">Eliminar</button></div></article>`).join("") || "<p>Guarda solves anisotrópicos de fotogramas fijos para iniciar el experimento.</p>";
    cameraStateList.querySelectorAll("[data-camera-state-reference]").forEach((button) => button.addEventListener("click", () => setCameraStateReference(button.dataset.cameraStateReference)));
    cameraStateList.querySelectorAll("[data-camera-state-remove]").forEach((button) => button.addEventListener("click", () => removeHumanState(button.dataset.cameraStateRemove)));
    cameraStateAnalysisElement.hidden = !cameraStateAnalysis;
    cameraStateAnalysisElement.innerHTML = analysisMarkup(cameraStateAnalysis);
  }

  function resetSolution() {
    projected = null; cameraModelProbe = null; currentSolve = null; projectionVisible = true; cameraModel = "pure_homography"; diagnosticFocus = null; projectionToggle.hidden = true; cameraModelModes.hidden = true; resultSection.hidden = true; result.textContent = "";
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
    updateUI();
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
    const sourceSelected = Boolean(selected?.playbackUrl && testVideo.src);
    const metadataReady = sourceSelected && testVideo.readyState >= HTMLMediaElement.HAVE_METADATA;
    const durationReady = metadataReady && (Number.isFinite(testVideo.duration) || Number(selected?.duration) > 0);
    const frameDataReady = metadataReady && testVideo.videoWidth > 0 && testVideo.videoHeight > 0 && testVideo.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA;
    const fpsReady = metadataReady && Boolean(selected?.fps);
    const navigationAllowed = !frameLocked;
    previousButton.disabled = !fpsReady || !navigationAllowed; nextButton.disabled = !fpsReady || !navigationAllowed; frameInput.disabled = !fpsReady || !navigationAllowed;
    playButton.disabled = !metadataReady || !navigationAllowed; timeline.disabled = !durationReady || !navigationAllowed; lockButton.disabled = !frameDataReady;
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
    document.querySelector("#test-undo").disabled = !frameLocked || (!history.length && !draft?.points.length);
    document.querySelector("#test-remove").disabled = !frameLocked || !selectedName || !selectedAnnotations.has(selectedName);
    document.querySelector("#test-clear").disabled = !frameLocked;
    document.querySelector("#test-remove").textContent = mode === "line" ? "Eliminar línea" : "Eliminar punto";
    document.querySelector("#test-solve").disabled = !frameLocked;
    updateControls();
    renderCameraStateExperiment();
  }
  function drawPolyline(points, closed) {
    if (!points?.length) return;
    context.beginPath(); context.moveTo(points[0][0], points[0][1]); points.slice(1).forEach(([x, y]) => context.lineTo(x, y)); if (closed) context.closePath(); context.stroke();
  }
  function draw() {
    if (!frameLocked || !canvas.width || !canvas.height) return;
    context.clearRect(0, 0, canvas.width, canvas.height); context.lineJoin = "round"; context.lineCap = "round";
    if (projected && projectionVisible) {
      drawProjectedGeometry();
    }
    const scale = Math.max(2, canvas.width / 900); context.strokeStyle = "#f2c94c"; context.fillStyle = "#f2c94c";
    lineAnnotations.forEach((annotation, name) => { const emphasis = (mode === "line" && name === selectedName) || (diagnosticFocus?.kind === "line" && diagnosticFocus.name === name); context.lineWidth = scale * (emphasis ? 1.75 : 1); context.beginPath(); context.moveTo(...annotation.points[0]); context.lineTo(...annotation.points[1]); context.stroke(); annotation.points.forEach(([x, y]) => { context.beginPath(); context.arc(x, y, context.lineWidth * 1.7, 0, Math.PI * 2); context.fill(); }); });
    pointAnnotations.forEach((annotation, name) => { const [x, y] = annotation.point; const emphasis = (mode === "point" && name === selectedName) || (diagnosticFocus?.kind === "point" && diagnosticFocus.name === name); context.beginPath(); context.arc(x, y, scale * (emphasis ? 2.8 : 1.9), 0, Math.PI * 2); context.fill(); });
    if (draft?.points.length) { const [x, y] = draft.points[0]; context.beginPath(); context.arc(x, y, scale * 2.4, 0, Math.PI * 2); context.fill(); }
  }
  function drawProjectedGeometry() {
    context.lineWidth = Math.max(1.5, canvas.width / 1150); context.strokeStyle = "rgba(103, 216, 255, .92)";
    const geometryToDraw = cameraModelProbe?.models?.[cameraModel]?.projected_geometry || projected;
    geometryToDraw.lines.forEach((line) => drawPolyline(line.points, false)); geometryToDraw.curves.forEach((curve) => drawPolyline(curve.points, curve.closed));
    context.fillStyle = "rgba(103, 216, 255, .92)"; geometryToDraw.points.forEach((point) => { context.beginPath(); context.arc(point.point[0], point.point[1], context.lineWidth * 1.2, 0, Math.PI * 2); context.fill(); });
  }
  function finite(value) { return typeof value === "number" && Number.isFinite(value); }
  function metricText(value) { return finite(value) ? `${value.toFixed(2)} px` : "—"; }
  function qualityClass(value) { return !finite(value) ? "" : value <= 5 ? "is-good" : value <= 15 ? "is-warning" : "is-danger"; }
  function labelFor(kind, name) { return geometry?.[kind === "line" ? "lines" : "points"]?.find((item) => item.name === name)?.label || name; }
  function metricBlock(title, metrics) {
    if (!metrics || ![metrics.median_px, metrics.p95_px, metrics.max_px].some(finite)) return "";
    return `<section class="test-diagnostic-block"><h3>${title}</h3><div class="test-diagnostic-metrics"><span>Mediana <strong>${metricText(metrics.median_px)}</strong></span><span>P95 <strong>${metricText(metrics.p95_px)}</strong></span><span>Máximo <strong>${metricText(metrics.max_px)}</strong></span></div></section>`;
  }
  function diagnosticRows(kind, values) {
    return Object.entries(values || {}).map(([name, value]) => {
      const metrics = kind === "line" ? value : { median_px: value, p95_px: null, max_px: value };
      const primary = kind === "line" ? metrics.median_px : value;
      return `<button class="test-diagnostic-row ${qualityClass(primary)}" type="button" data-diagnostic-kind="${kind}" data-diagnostic-name="${name}"><span><b>${labelFor(kind, name)}</b><small>${kind === "line" ? `Mediana ${metricText(metrics.median_px)} · P95 ${metricText(metrics.p95_px)}` : metricText(value)}</small></span><i aria-hidden="true"></i></button>`;
    }).join("");
  }
  function renderDiagnostics(solved) {
    const metrics = solved.metrics || {};
    const support = solved.projection_support || {};
    const probe = solved.camera_model_probe || {};
    const pointRows = diagnosticRows("point", metrics.per_point);
    const lineRows = diagnosticRows("line", metrics.per_line);
    const pointBlock = Object.keys(metrics.per_point || {}).length ? `<section class="test-diagnostic-block"><h3>Puntos</h3><div class="test-detail-list">${pointRows}</div></section>` : "";
    const lineBlock = Object.keys(metrics.per_line || {}).length ? `<section class="test-diagnostic-block"><h3>Líneas</h3><div class="test-detail-list">${lineRows}</div></section>` : "";
    const supportRatio = finite(support.support_ratio_of_visible) ? `${(support.support_ratio_of_visible * 100).toFixed(0)} %` : "—";
    const supportWarnings = (support.warnings || []).map((warning) => `<li>${warning}</li>`).join("");
    const supportStatus = support.projective_valid === false ? "⚠ Revisar geometría proyectiva" : "Correcta";
    const modelLabel = { pure_homography: "Homografía", radial_1p: "Radial 1P", anisotropic_2p: "Anisotrópica 2P" };
    const modelRows = Object.entries(probe.models || {}).map(([name, model]) => {
      const params = model.parameters || [];
      const parameters = name === "radial_1p" ? `k ${finite(params[0]) ? params[0].toFixed(4) : "—"}` : name === "anisotropic_2p" ? `kx ${finite(params[0]) ? params[0].toFixed(4) : "—"} · ky ${finite(params[1]) ? params[1].toFixed(4) : "—"}` : "H sin parámetros adicionales";
      const summary = model.residual_summary || {};
      const improvement = finite(model.annotation_objective_improvement_percent) ? ` · Mejora ${model.annotation_objective_improvement_percent.toFixed(1)} %` : "";
      const warp = finite(model.max_warp_px) ? `Warp máx. ${model.max_warp_px.toFixed(1)} px` : "Warp máx. —";
      return `<div class="test-camera-model-row"><strong>${modelLabel[name] || name}</strong><small>${parameters}</small><span>Mediana ${metricText(summary.median_px)} · P95 ${metricText(summary.p95_px)}${improvement}</span><small>${warp}</small></div>`;
    }).join("");
    const bestLabel = modelLabel[probe.best_regularized_model] || probe.best_regularized_model || "—";
    const cameraWarning = probe.warning ? `<p class="test-support-warnings">${probe.warning}</p>` : "";
    result.innerHTML = `<div class="test-diagnostic-summary"><h3>Evidencia</h3><dl><div><dt>Líneas</dt><dd>${solved.line_annotation_count ?? 0}</dd></div><div><dt>Puntos</dt><dd>${solved.point_annotation_count ?? 0}</dd></div><div><dt>Restricciones de línea</dt><dd>${solved.line_constraint_count ?? 0}</dd></div><div><dt>Restricciones de punto</dt><dd>${solved.direct_correspondence_count ?? 0}</dd></div><div><dt>Rango geométrico</dt><dd>${solved.linear_rank ?? "—"} / 8</dd></div><div><dt>Refinamiento</dt><dd>${solved.refinement_iterations ?? "—"} iteraciones</dd></div></dl></div><section class="test-diagnostic-block"><h3>Error global</h3><p class="test-diagnostic-caption">Error balanceado de las anotaciones</p><div class="test-diagnostic-metrics"><span>Mediana <strong>${metricText(metrics.median_px)}</strong></span><span>P95 <strong>${metricText(metrics.p95_px)}</strong></span><span>Máximo <strong>${metricText(metrics.max_px)}</strong></span></div></section>${metricBlock("Error de puntos", metrics.point_error)}${metricBlock("Error de líneas", metrics.line_error)}<section class="test-diagnostic-block"><h3>Modelo de cámara</h3><p class="test-diagnostic-caption">Comparación experimental sobre las mismas anotaciones. Un error menor no demuestra por sí solo que el modelo generalice.</p><div class="test-camera-model-list">${modelRows || "Sin datos del probe."}</div>${probe.models ? `<p class="test-diagnostic-caption">Mejor ajuste regularizado: ${bestLabel}</p>` : ""}</section><section class="test-diagnostic-block"><h3>Soporte de proyección</h3><dl><div><dt>Puntos de soporte</dt><dd>${support.point_evidence_count ?? 0}</dd></div><div><dt>Líneas de soporte</dt><dd>${support.line_evidence_count ?? 0}</dd></div><div><dt>Cobertura visible</dt><dd>${supportRatio}</dd></div><div><dt>Margen</dt><dd>${finite(support.support_margin_m) ? `${support.support_margin_m.toFixed(1)} m` : "—"}</dd></div><div><dt>Geometría proyectiva</dt><dd>${supportStatus}</dd></div></dl>${supportWarnings ? `<ul class="test-support-warnings">${supportWarnings}</ul>` : ""}</section>${cameraWarning}<section class="test-diagnostic-block"><h3>Detalle de anotaciones</h3>${pointBlock}${lineBlock}</section>`;
    result.querySelectorAll("[data-diagnostic-name]").forEach((row) => {
      row.addEventListener("mouseenter", () => { diagnosticFocus = { kind: row.dataset.diagnosticKind, name: row.dataset.diagnosticName }; draw(); });
      row.addEventListener("mouseleave", () => { diagnosticFocus = null; draw(); });
      row.addEventListener("focus", () => { diagnosticFocus = { kind: row.dataset.diagnosticKind, name: row.dataset.diagnosticName }; draw(); });
      row.addEventListener("blur", () => { diagnosticFocus = null; draw(); });
    });
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
    if (!selected) return;
    if (testVideo.readyState < HTMLMediaElement.HAVE_CURRENT_DATA || !testVideo.videoWidth || !testVideo.videoHeight) {
      frameInfo.textContent = "Cargando fotograma…";
      updateControls();
      return;
    }
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
    if (sourceChanged && cameraStateVideoAssetId && cameraStateVideoAssetId !== selected.assetId) resetCameraStateExperiment();
    pendingStartTime = sourceVideo?.currentTime || 0;
    if (sourceChanged) { unlockFrame(); testVideo.dataset.assetId = selected.assetId; testVideo.src = selected.playbackUrl; testVideo.load(); setEmpty("Cargando vídeo…"); }
    else if (testVideo.readyState >= HTMLMediaElement.HAVE_METADATA) { if (!frameLocked && pendingStartTime !== null) { testVideo.currentTime = pendingStartTime; pendingStartTime = null; } empty.hidden = true; updatePlayback(); updateUI(); }
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
      currentSolve = solved; projected = solved.projected_geometry; cameraModelProbe = solved.camera_model_probe || null; projectionVisible = true; cameraModel = "pure_homography"; projectionToggle.hidden = false; cameraModelModes.hidden = !Object.keys(cameraModelProbe?.models || {}).length; homographyButton.classList.add("is-active"); radialButton.classList.remove("is-active"); anisotropicButton.classList.remove("is-active"); cameraBest.textContent = cameraModelProbe?.best_regularized_model ? `Mejor ajuste: ${{ pure_homography: "Homografía", radial_1p: "Radial 1P", anisotropic_2p: "Anisotrópica 2P" }[cameraModelProbe.best_regularized_model] || cameraModelProbe.best_regularized_model}` : ""; diagnosticFocus = null; renderDiagnostics(solved);
      updateUI(); draw();
    } catch (error) { resultSection.hidden = false; result.textContent = `No se pudo resolver: ${error.message}`; }
  });
  projectionToggle.addEventListener("click", () => { projectionVisible = !projectionVisible; projectionToggle.textContent = projectionVisible ? "Ocultar proyección" : "Mostrar proyección"; draw(); });
  homographyButton.addEventListener("click", () => { cameraModel = "pure_homography"; homographyButton.classList.add("is-active"); radialButton.classList.remove("is-active"); anisotropicButton.classList.remove("is-active"); draw(); });
  radialButton.addEventListener("click", () => { cameraModel = "radial_1p"; radialButton.classList.add("is-active"); homographyButton.classList.remove("is-active"); anisotropicButton.classList.remove("is-active"); draw(); });
  anisotropicButton.addEventListener("click", () => { cameraModel = "anisotropic_2p"; anisotropicButton.classList.add("is-active"); homographyButton.classList.remove("is-active"); radialButton.classList.remove("is-active"); draw(); });
  saveHumanButton.addEventListener("click", saveCurrentHumanState);
  resetCameraStatesButton.addEventListener("click", resetCameraStateExperiment);
  analyzeCameraStatesButton.addEventListener("click", async () => {
    if (humanStates.length < 2 || !cameraStateReferenceId) return;
    analyzeCameraStatesButton.disabled = true;
    cameraStateAnalysisElement.hidden = false;
    cameraStateAnalysisElement.textContent = "Analizando relaciones entre estados HUMAN…";
    try {
      cameraStateAnalysis = await api("/api/field/camera-state/probe", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          projection_model: "anisotropic_2p",
          reference_state_id: cameraStateReferenceId,
          states: humanStates,
        }),
      });
    } catch (error) {
      cameraStateAnalysis = null;
      cameraStateAnalysisElement.hidden = false;
      cameraStateAnalysisElement.textContent = `No se pudo analizar el experimento: ${error.message}`;
      analyzeCameraStatesButton.disabled = false;
      return;
    }
    renderCameraStateExperiment();
  });
  previousButton.addEventListener("click", () => stepFrame(-1)); nextButton.addEventListener("click", () => stepFrame(1));
  playButton.addEventListener("click", () => { if (testVideo.paused) void testVideo.play(); else testVideo.pause(); });
  timeline.addEventListener("input", () => { unlockFrame(); testVideo.currentTime = Number(timeline.value); });
  frameInput.addEventListener("keydown", (event) => { if (event.key === "Enter") { event.preventDefault(); seekFrame(Number(frameInput.value)); } }); frameInput.addEventListener("change", () => seekFrame(Number(frameInput.value)));
  lockButton.addEventListener("click", () => { if (frameLocked) unlockFrame(); else lockFrame(); });
  testVideo.addEventListener("loadedmetadata", () => { if (pendingStartTime !== null && !frameLocked) { testVideo.currentTime = pendingStartTime; pendingStartTime = null; } empty.hidden = true; updatePlayback(); updateUI(); });
  testVideo.addEventListener("loadeddata", () => { empty.hidden = true; updatePlayback(); updateUI(); });
  testVideo.addEventListener("canplay", () => { updatePlayback(); updateUI(); });
  testVideo.addEventListener("durationchange", () => { updatePlayback(); updateUI(); });
  testVideo.addEventListener("timeupdate", () => { updatePlayback(); updateUI(); }); testVideo.addEventListener("seeked", () => { updatePlayback(); updateUI(); });
  testVideo.addEventListener("play", () => { unlockFrame(); updateUI(); }); testVideo.addEventListener("pause", updateUI);
  testVideo.addEventListener("error", () => { setEmpty("No se pudo cargar el vídeo seleccionado."); updateUI(); });
  window.addEventListener("resize", draw);
  window.SupervisorTest = { enter: initialiseVideo, leave: () => { const sourceVideo = window.SupervisorWorkspace?.getVideo(); if (sourceVideo && testVideo.readyState >= 1) sourceVideo.currentTime = testVideo.currentTime; } };
  api("/api/field/pitch-lines").then((data) => { geometry = data; renderMiniPitch(); updateUI(); }).catch((error) => { resultSection.hidden = false; result.textContent = error.message; });
})();
