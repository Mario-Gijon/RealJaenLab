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
  const evaluateCameraStateCoverageButton = document.querySelector("#test-evaluate-camera-state-coverage");
  const analyzeCameraStatesButton = document.querySelector("#test-analyze-camera-states");
  const resetCameraStatesButton = document.querySelector("#test-reset-camera-states");
  const cameraStateList = document.querySelector("#test-camera-state-list");
  const humanAtlasElement = document.querySelector("#test-human-atlas");
  const cameraStateCoverageElement = document.querySelector("#test-camera-state-coverage");
  const cameraStateAnalysisElement = document.querySelector("#test-camera-state-analysis");
  const propagateHumanButton = document.querySelector("#test-propagate-human");
  const viewTransformModes = document.querySelector("#test-view-transform-modes");
  const viewTransformResult = document.querySelector("#test-view-transform-result");
  const refineDerivedButton = document.querySelector("#test-refine-derived");
  const saveAssistedHumanButton = document.querySelector("#test-save-assisted-human");
  const derivedRefinementModes = document.querySelector("#test-derived-refinement-modes");
  const derivedRefinementResult = document.querySelector("#test-derived-refinement-result");
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
  let humanAtlasSummary = null;
  let humanAtlasError = null;
  let nextHumanDescriptorRequestId = 0;
  let humanAtlasRequestId = 0;
  let cameraStateCoverageTarget = null;
  let cameraStateCoverageError = null;
  let cameraStateCoverageRequestId = 0;
  let cameraStateCoverageRunning = false;
  let viewTransformPropagation = null;
  let viewTransformModel = null;
  let derivedRefinement = null;
  let derivedRefinementModel = null;
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
  const humanGeometry = (state) => state?.projected_geometry
    || state?.solve?.camera_model_probe?.models?.anisotropic_2p?.projected_geometry
    || null;
  const isDirectHuman = (state) => state?.anchor_type !== "assisted" && Boolean(state?.solve);
  const directHumanStates = () => humanStates.filter(isDirectHuman);
  const referenceHuman = () => humanStates.find((state) => state.state_id === cameraStateReferenceId) || null;
  const referenceHumanGeometry = () => humanGeometry(referenceHuman());
  const propagationReady = () => {
    const selected = window.SupervisorWorkspace?.getSelectedVideo();
    const reference = referenceHuman();
    return Boolean(selected?.assetId && frameLocked && Number.isInteger(fixedFrame)
      && reference && cameraStateVideoAssetId === selected.assetId
      && reference.frame_index !== fixedFrame && referenceHumanGeometry());
  };
  const selectedTransportedGeometry = () => viewTransformPropagation?.models?.[viewTransformModel]?.transported_geometry || null;
  const refinementReady = () => Boolean(
    frameLocked && viewTransformPropagation
    && viewTransformPropagation.target_frame_index === fixedFrame
    && selectedTransportedGeometry()
    && (lineAnnotations.size + pointAnnotations.size) > 0
  );
  const selectedRefinedGeometry = () => derivedRefinementModel && derivedRefinementModel !== "propagated"
    ? derivedRefinement?.models?.[derivedRefinementModel]?.refined_geometry || null
    : null;
  const matchesExperimentField = (state) => !state?.field_profile
    || (Math.abs(Number(state.field_profile.length) - Number(cameraStateFieldProfile?.length)) < 1e-6
      && Math.abs(Number(state.field_profile.width) - Number(cameraStateFieldProfile?.width)) < 1e-6);
  const assistedHumanReady = () => {
    const selected = window.SupervisorWorkspace?.getSelectedVideo();
    const source = referenceHuman();
    return Boolean(selected?.assetId && frameLocked && Number.isInteger(fixedFrame)
      && cameraStateFieldProfile && cameraStateVideoAssetId === selected.assetId
      && source && viewTransformPropagation?.source_frame_index === source.frame_index
      && viewTransformPropagation?.target_frame_index === fixedFrame
      && fixedFrame !== source.frame_index && matchesExperimentField(source)
      && selectedRefinedGeometry());
  };
  const cameraStateCoverageAnchors = () => humanStates.filter((state) => (
    Number.isInteger(state.frame_index)
    && humanGeometry(state)
    && matchesExperimentField(state)
  ));
  const cameraStateCoverageReady = () => {
    const selected = window.SupervisorWorkspace?.getSelectedVideo();
    return Boolean(selected?.assetId && frameLocked && Number.isInteger(fixedFrame)
      && cameraStateFieldProfile && cameraStateVideoAssetId === selected.assetId
      && cameraStateCoverageAnchors().some((state) => state.frame_index !== fixedFrame));
  };

  function clearCameraStateAnalysis() {
    cameraStateAnalysis = null;
  }
  function invalidateCameraStateCoverageTarget() {
    cameraStateCoverageRequestId += 1;
    cameraStateCoverageTarget = null;
    cameraStateCoverageError = null;
    cameraStateCoverageRunning = false;
  }
  function clearViewTransformPropagation() {
    viewTransformPropagation = null;
    viewTransformModel = null;
    clearDerivedRefinement();
  }
  function clearDerivedRefinement() {
    derivedRefinement = null;
    derivedRefinementModel = null;
  }
  function resetCameraStateExperiment() {
    humanStates = [];
    cameraStateReferenceId = null;
    cameraStateAnalysis = null;
    cameraStateFieldProfile = null;
    cameraStateVideoAssetId = null;
    humanAtlasSummary = null;
    humanAtlasError = null;
    humanAtlasRequestId += 1;
    invalidateCameraStateCoverageTarget();
    clearViewTransformPropagation();
    renderCameraStateExperiment();
  }
  function validHumanDescriptors() {
    return humanStates.filter((state) => state.view_descriptor && typeof state.view_descriptor === "object");
  }
  function refreshHumanAtlas() {
    const requestId = ++humanAtlasRequestId;
    const descriptorStates = validHumanDescriptors();
    const descriptors = descriptorStates.map((state) => state.view_descriptor);
    humanAtlasSummary = null;
    humanAtlasError = null;
    if (!descriptors.length) return;
    void api("/api/field/human-atlas/summarize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ descriptors }),
    }).then((summary) => {
      if (requestId !== humanAtlasRequestId) return;
      const current = validHumanDescriptors();
      if (current.length !== descriptorStates.length || current.some((state, index) => state.view_descriptor !== descriptors[index])) return;
      humanAtlasSummary = summary;
      renderCameraStateExperiment();
    }).catch((error) => {
      if (requestId !== humanAtlasRequestId) return;
      const current = validHumanDescriptors();
      if (current.length !== descriptorStates.length || current.some((state, index) => state.view_descriptor !== descriptors[index])) return;
      humanAtlasError = error.message;
      renderCameraStateExperiment();
    });
  }
  function requestHumanViewDescriptor(state) {
    const videoWidth = testVideo.videoWidth || canvas.width;
    const videoHeight = testVideo.videoHeight || canvas.height;
    const requestId = ++nextHumanDescriptorRequestId;
    state.view_descriptor = null;
    state.view_descriptor_error = null;
    state.view_descriptor_request_id = requestId;
    if (!finite(videoWidth) || !finite(videoHeight) || videoWidth <= 0 || videoHeight <= 0) {
      state.view_descriptor_error = "No se conocen las dimensiones reales del vídeo.";
      refreshHumanAtlas();
      return;
    }
    void api("/api/field/human-view/describe", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        projected_geometry: cloneJson(humanGeometry(state)),
        field_profile: cloneJson(state.field_profile),
        video_width: videoWidth,
        video_height: videoHeight,
        human_id: state.state_id,
      }),
    }).then((descriptor) => {
      const current = humanStates.find((candidate) => candidate.state_id === state.state_id);
      if (!current || current.view_descriptor_request_id !== requestId) return;
      current.view_descriptor = descriptor;
      current.view_descriptor_error = null;
      invalidateCameraStateCoverageTarget();
      refreshHumanAtlas();
      renderCameraStateExperiment();
    }).catch((error) => {
      const current = humanStates.find((candidate) => candidate.state_id === state.state_id);
      if (!current || current.view_descriptor_request_id !== requestId) return;
      current.view_descriptor = null;
      current.view_descriptor_error = error.message;
      invalidateCameraStateCoverageTarget();
      refreshHumanAtlas();
      renderCameraStateExperiment();
    });
  }
  function upsertHumanState(state) {
    const existing = humanStates.findIndex((candidate) => candidate.state_id === state.state_id);
    if (existing >= 0) humanStates.splice(existing, 1, state); else humanStates.push(state);
    invalidateCameraStateCoverageTarget();
    requestHumanViewDescriptor(state);
    refreshHumanAtlas();
  }
  function saveCurrentHumanState() {
    if (!frameLocked || !Number.isInteger(fixedFrame) || !anisotropicGeometryAvailable()) return;
    const stateId = `frame_${fixedFrame}`;
    const solve = cloneJson(currentSolve);
    const state = {
      state_id: stateId,
      frame_index: fixedFrame,
      timestamp_seconds: fixedTimestamp,
      anchor_type: "direct",
      projection_model: "anisotropic_2p",
      projected_geometry: solve.camera_model_probe.models.anisotropic_2p.projected_geometry,
      field_profile: { length: solve.pitch_length, width: solve.pitch_width },
      provenance: { kind: "independent_solve", projection_model: "anisotropic_2p" },
      solve,
    };
    upsertHumanState(state);
    if (!cameraStateFieldProfile) {
      cameraStateFieldProfile = {
        length: currentSolve.pitch_length,
        width: currentSolve.pitch_width,
      };
    }
    cameraStateVideoAssetId = window.SupervisorWorkspace?.getSelectedVideo()?.assetId || null;
    if (!cameraStateReferenceId) cameraStateReferenceId = stateId;
    clearCameraStateAnalysis();
    clearViewTransformPropagation();
    renderCameraStateExperiment();
  }
  function saveAssistedHumanState() {
    if (!assistedHumanReady()) return;
    const selected = window.SupervisorWorkspace?.getSelectedVideo();
    const source = referenceHuman();
    const refinedGeometry = selectedRefinedGeometry();
    const stateId = `frame_${fixedFrame}`;
    const refinementModel = derivedRefinement.models[derivedRefinementModel];
    const baselineP95 = Number(derivedRefinement.baseline?.residual_summary?.p95_px);
    const selectedP95 = Number(refinementModel?.residual_summary?.p95_px);
    const p95ImprovementPercent = finite(baselineP95) && finite(selectedP95) && baselineP95 > 0
      ? 100 * (baselineP95 - selectedP95) / baselineP95
      : null;
    const state = {
      state_id: stateId,
      frame_index: fixedFrame,
      timestamp_seconds: fixedTimestamp,
      anchor_type: "assisted",
      projection_model: "anisotropic_2p",
      projected_geometry: cloneJson(refinedGeometry),
      field_profile: cloneJson(cameraStateFieldProfile),
      provenance: {
        kind: "derived_refinement",
        source_human_state_id: source.state_id,
        source_human_frame_index: source.frame_index,
        view_transform_model: viewTransformModel,
        refinement_model: derivedRefinementModel,
        baseline_residual_summary: cloneJson(derivedRefinement.baseline?.residual_summary || {}),
        refined_residual_summary: cloneJson(refinementModel?.residual_summary || {}),
        p95_improvement_percent: p95ImprovementPercent,
      },
    };
    upsertHumanState(state);
    cameraStateVideoAssetId = selected.assetId;
    if (!cameraStateReferenceId) cameraStateReferenceId = stateId;
    clearCameraStateAnalysis();
    renderCameraStateExperiment();
  }
  function setCameraStateReference(stateId) {
    if (!humanStates.some((state) => state.state_id === stateId)) return;
    cameraStateReferenceId = stateId;
    clearCameraStateAnalysis();
    clearViewTransformPropagation();
    renderCameraStateExperiment();
  }
  function removeHumanState(stateId) {
    humanStates = humanStates.filter((state) => state.state_id !== stateId);
    invalidateCameraStateCoverageTarget();
    if (cameraStateReferenceId === stateId) cameraStateReferenceId = humanStates[0]?.state_id || null;
    if (!humanStates.length) {
      cameraStateFieldProfile = null;
      cameraStateVideoAssetId = null;
    }
    refreshHumanAtlas();
    clearCameraStateAnalysis();
    clearViewTransformPropagation();
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
  function viewTransformMarkup(propagation) {
    if (!propagation) return "";
    const selected = propagation.models?.[viewTransformModel] || {};
    const modelRows = ["translation", "similarity", "affine", "homography"].map((name) => {
      const model = propagation.models?.[name] || {};
      const metrics = model.residual_summary || {};
      const recommended = name === propagation.recommended_model;
      return `<div class="test-camera-state-model ${recommended ? "is-recommended" : ""}"><strong>${cameraStateModelLabel(name)}${recommended ? " · recomendado" : ""}</strong><br>Mediana ${metricText(metrics.median_px)} · P95 ${metricText(metrics.p95_px)} · Máximo ${metricText(metrics.max_px)}</div>`;
    }).join("");
    const warnings = (propagation.warnings || []).map((warning) => `<p class="test-camera-state-warning">${warning}</p>`).join("");
    return `<h3>Diagnóstico DERIVED</h3><dl><div><dt>HUMAN origen</dt><dd>Fotograma ${propagation.source_frame_index ?? "—"}</dd></div><div><dt>Destino</dt><dd>Fotograma ${propagation.target_frame_index ?? "—"}</dd></div><div><dt>Features</dt><dd>${propagation.feature_method || "—"}</dd></div><div><dt>Keypoints origen/destino</dt><dd>${propagation.source_keypoint_count ?? "—"} / ${propagation.target_keypoint_count ?? "—"}</dd></div><div><dt>Matches mutuos</dt><dd>${propagation.mutual_match_count ?? "—"}</dd></div><div><dt>Inliers robustos</dt><dd>${propagation.consensus_inlier_count ?? "—"} · ${finite(propagation.consensus_inlier_ratio) ? `${(propagation.consensus_inlier_ratio * 100).toFixed(1)} %` : "—"}</dd></div><div><dt>Cobertura origen/destino</dt><dd>${finite(propagation.source_spatial_coverage) ? `${(propagation.source_spatial_coverage * 100).toFixed(1)} %` : "—"} / ${finite(propagation.target_spatial_coverage) ? `${(propagation.target_spatial_coverage * 100).toFixed(1)} %` : "—"}</dd></div><div><dt>Recomendado</dt><dd>${cameraStateModelLabel(propagation.recommended_model)}</dd></div></dl>${warnings}<div class="test-camera-state-models">${modelRows}</div><p class="test-diagnostic-caption">${cameraStateModelLabel(viewTransformModel)} seleccionado. ${transformDetails(selected.transform)}</p>`;
  }
  function renderViewTransform() {
    propagateHumanButton.disabled = !propagationReady();
    if (!projected && !viewTransformPropagation) projectionToggle.hidden = true;
    else if (!projected && viewTransformPropagation) {
      projectionToggle.hidden = false;
      projectionToggle.textContent = projectionVisible ? "Ocultar proyección" : "Mostrar proyección";
    }
    viewTransformModes.hidden = !viewTransformPropagation;
    viewTransformModes.querySelectorAll("[data-view-transform-model]").forEach((button) => {
      const name = button.dataset.viewTransformModel;
      button.classList.toggle("is-active", name === viewTransformModel);
      button.classList.toggle("is-recommended", name === viewTransformPropagation?.recommended_model);
    });
    viewTransformResult.hidden = !viewTransformPropagation;
    viewTransformResult.innerHTML = viewTransformMarkup(viewTransformPropagation);
  }
  function derivedRefinementMarkup(refinement) {
    if (!refinement) return "";
    const baseline = refinement.baseline?.residual_summary || {};
    const recommended = refinement.models?.[refinement.recommended_model] || {};
    const recommendedSummary = recommended.residual_summary || {};
    const selected = derivedRefinementModel === "propagated" ? refinement.baseline : refinement.models?.[derivedRefinementModel];
    const details = selected?.annotation_details || refinement.baseline?.annotation_details || [];
    const modelRows = ["translation", "similarity", "affine", "homography"].map((name) => {
      const model = refinement.models?.[name] || {};
      if (model.error) return `<div class="test-camera-state-model"><strong>${cameraStateModelLabel(name)}</strong> · No disponible: ${model.error}</div>`;
      const summary = model.residual_summary || {};
      return `<div class="test-camera-state-model ${name === refinement.recommended_model ? "is-recommended" : ""}"><strong>${cameraStateModelLabel(name)}${name === refinement.recommended_model ? " · recomendado" : ""}</strong><br>Mediana ${metricText(summary.median_px)} · P95 ${metricText(summary.p95_px)} · Máximo ${metricText(summary.max_px)}<br>Regularización ${numberText(model.regularization)} · pérdida ${numberText(model.data_loss)}</div>`;
    }).join("");
    const annotationDetails = details.map((detail) => detail.kind === "point"
      ? `<div class="test-derived-refinement-detail"><strong>Punto ${detail.name}</strong><small>Error ${metricText(detail.error_px)}</small></div>`
      : `<div class="test-derived-refinement-detail"><strong>Línea ${detail.name}</strong><small>Clicks ${detail.click_errors_px?.map((value) => metricText(value)).join(" · ") || "—"} · mediana ${metricText(detail.median_px)}</small></div>`).join("");
    const warnings = (refinement.warnings || []).map((warning) => `<p class="test-camera-state-warning">${warning}</p>`).join("");
    return `<h3>DERIVED · Diagnóstico refinado</h3><dl><div><dt>Prior</dt><dd>${cameraStateModelLabel(viewTransformModel)}</dd></div><div><dt>Puntos / líneas</dt><dd>${refinement.point_annotation_count ?? 0} / ${refinement.line_annotation_count ?? 0}</dd></div><div><dt>Restricciones escalares</dt><dd>${refinement.constraint_count ?? 0}</dd></div><div><dt>Corrección recomendada</dt><dd>${cameraStateModelLabel(refinement.recommended_model)}</dd></div><div><dt>Baseline mediana / P95 / máx.</dt><dd>${metricText(baseline.median_px)} · ${metricText(baseline.p95_px)} · ${metricText(baseline.max_px)}</dd></div><div><dt>Refinado mediana / P95 / máx.</dt><dd>${metricText(recommendedSummary.median_px)} · ${metricText(recommendedSummary.p95_px)} · ${metricText(recommendedSummary.max_px)}</dd></div><div><dt>Mejora P95</dt><dd>${numberText(refinement.recommended_p95_improvement_percent, 1)} %</dd></div></dl>${warnings}<div class="test-camera-state-models">${modelRows}</div>${selected?.transform ? `<p class="test-diagnostic-caption">${cameraStateModelLabel(derivedRefinementModel)} seleccionado. ${transformDetails(selected.transform)}</p>` : ""}${annotationDetails ? `<h3>Detalle de anotaciones</h3><div class="test-derived-refinement-details">${annotationDetails}</div>` : ""}`;
  }
  function humanDescriptorMarkup(state) {
    const descriptor = state.view_descriptor;
    if (!descriptor) {
      if (state.view_descriptor_error) return `<p class="test-camera-state-warning">FIELD no disponible: ${state.view_descriptor_error}</p>`;
      return `<small>Cobertura FIELD: calculando descriptor…</small>`;
    }
    const coverage = descriptor.coverage || {};
    const center = coverage.view_center_normalized || [];
    const scale = descriptor.effective_view_scale || {};
    const visibility = descriptor.semantic_visibility || {};
    const approximation = descriptor.approximation?.residual_summary || {};
    const warnings = (descriptor.warnings || []).map((warning) => `⚠ ${warning}`).join(" · ");
    return `<small>FIELD ${numberText(coverage.visible_field_percent, 1)}% · centro ${numberText(center[0])} / ${numberText(center[1])} · escala ${numberText(scale.scale_geometric_px_per_m, 1)} px/m<br>Landmarks ${visibility.visible_landmark_count ?? "—"}/${visibility.total_landmark_count ?? "—"} · approx P95 ${metricText(approximation.p95_px)}${warnings ? `<br>${warnings}` : ""}</small>`;
  }
  function humanAtlasMarkup() {
    const descriptorStates = validHumanDescriptors();
    if (!humanStates.length) return "";
    if (!descriptorStates.length) {
      const unavailable = humanStates.some((state) => state.view_descriptor_error)
        ? "Cobertura FIELD no disponible para los HUMAN sin descriptor válido."
        : "Calculando descriptores HUMAN para la cobertura FIELD…";
      return `<h3>Human Atlas · Cobertura FIELD</h3><dl><div><dt>HUMAN total / descriptor válido</dt><dd>${humanStates.length} / 0</dd></div><div><dt>Cobertura FIELD acumulada</dt><dd>No disponible</dd></div><div><dt>Muestras cubiertas / total</dt><dd>— / —</dd></div></dl><p>${unavailable}</p><p class="test-diagnostic-caption">La cobertura FIELD mide qué superficie canónica del campo aparece en al menos un HUMAN. No mide todavía la cobertura de estados de cámara.</p>`;
    }
    if (humanAtlasError) {
      return `<h3>Human Atlas · Cobertura FIELD</h3><dl><div><dt>HUMAN total / descriptor válido</dt><dd>${humanStates.length} / ${descriptorStates.length}</dd></div><div><dt>Cobertura FIELD acumulada</dt><dd>No disponible</dd></div><div><dt>Muestras cubiertas / total</dt><dd>— / —</dd></div></dl><p class="test-camera-state-warning">Atlas no disponible: ${humanAtlasError}</p><p class="test-diagnostic-caption">La cobertura FIELD mide qué superficie canónica del campo aparece en al menos un HUMAN. No mide todavía la cobertura de estados de cámara.</p>`;
    }
    if (!humanAtlasSummary) {
      return `<h3>Human Atlas · Cobertura FIELD</h3><dl><div><dt>HUMAN total / descriptor válido</dt><dd>${humanStates.length} / ${descriptorStates.length}</dd></div><div><dt>Cobertura FIELD acumulada</dt><dd>Calculando…</dd></div><div><dt>Muestras cubiertas / total</dt><dd>— / —</dd></div></dl><p>Calculando cobertura acumulada de ${descriptorStates.length} HUMAN…</p><p class="test-diagnostic-caption">La cobertura FIELD mide qué superficie canónica del campo aparece en al menos un HUMAN. No mide todavía la cobertura de estados de cámara.</p>`;
    }
    const grid = humanAtlasSummary.grid || {};
    const marginal = (humanAtlasSummary.marginal_coverage || []).map((entry) => {
      const state = humanStates.find((candidate) => candidate.state_id === entry.human_id);
      return `<small>Frame ${state?.frame_index ?? entry.human_id} · +${numberText(entry.marginal_field_coverage_percent, 1)}% · acumulado ${numberText(entry.cumulative_field_coverage_percent, 1)}%</small>`;
    }).join("");
    return `<h3>Human Atlas · Cobertura FIELD</h3><dl><div><dt>HUMAN total / descriptor válido</dt><dd>${humanStates.length} / ${descriptorStates.length}</dd></div><div><dt>Cobertura FIELD acumulada</dt><dd>${numberText(humanAtlasSummary.accumulated_field_coverage_percent, 1)}%</dd></div><div><dt>Muestras cubiertas / total</dt><dd>${grid.covered_sample_count ?? "—"} / ${grid.total_sample_count ?? "—"}</dd></div></dl>${marginal}<p class="test-diagnostic-caption">La cobertura FIELD mide qué superficie canónica del campo aparece en al menos un HUMAN. No mide todavía la cobertura de estados de cámara.</p>`;
  }
  function cameraStateCoverageMarkup() {
    const heading = "<h3>Camera-State Coverage · Target</h3>";
    if (cameraStateCoverageRunning) {
      return `${heading}<p>Comparando target ${fixedFrame ?? "—"} con ${cameraStateCoverageAnchors().filter((state) => state.frame_index !== fixedFrame).length} HUMAN…</p>`;
    }
    if (cameraStateCoverageError) {
      return `${heading}<p class="test-camera-state-warning">No se pudo evaluar el target: ${cameraStateCoverageError}</p>`;
    }
    if (!cameraStateCoverageTarget) {
      return `${heading}<p>Evalúa si el fotograma fijado está representado por los HUMAN actuales. No necesita anotaciones del target.</p>`;
    }
    const coverage = cameraStateCoverageTarget;
    const best = coverage.best_candidate || {};
    const probe = best.probe_summary || {};
    const predicted = best.predicted_descriptor || {};
    const predictedCoverage = predicted.coverage || {};
    const predictedCenter = predictedCoverage.view_center_normalized || [];
    const predictedScale = predicted.effective_view_scale || {};
    const predictedApproximation = predicted.approximation?.residual_summary || {};
    const candidateById = new Map((coverage.candidates || []).map((candidate) => [candidate.anchor_id, candidate]));
    const ranking = (coverage.candidates || []).slice(0, 3).map((candidate) => `<small>Frame ${candidate.anchor_frame_index ?? candidate.anchor_id} · evidencia ${numberText(candidate.evidence_score)} · distancia ${numberText(candidate.state_distance)} · score ${numberText(candidate.candidate_score)}</small>`).join("");
    const consensus = coverage.coverage_mode === "multi_anchor_consensus" && coverage.consensus
      ? `<small>Consenso: ${(coverage.consensus.anchor_ids || []).map((id) => { const candidate = candidateById.get(id); return `frame ${candidate?.anchor_frame_index ?? id}`; }).join(" + ")} · desacuerdo ${numberText(coverage.consensus.prediction_disagreement)}</small><p class="test-diagnostic-caption">Dos predicciones HUMAN concuerdan sobre el estado del target; no se interpola ningún estado.</p>`
      : "";
    const failures = coverage.probe_failures?.length
      ? `<p class="test-camera-state-warning">${coverage.probe_failures.length} HUMAN no pudieron compararse visualmente.</p>`
      : "";
    const statusCopy = {
      covered: "El target está representado conservadoramente por el Atlas HUMAN.",
      uncertain: "El target está cerca del límite del Atlas; aún no cuenta como cobertura estricta.",
      uncovered: "Hay evidencia visual, pero ningún HUMAN representa este estado con suficiente confianza.",
      unobservable: "No hay evidencia visual suficiente para decidir si falta un estado.",
    }[coverage.status] || "Diagnóstico sin clasificación.";
    const predictedMarkup = best.predicted_descriptor
      ? `<h3>Predicción transportada</h3><dl><div><dt>Centro x / y</dt><dd>${numberText(predictedCenter[0])} / ${numberText(predictedCenter[1])}</dd></div><div><dt>Escala geométrica</dt><dd>${numberText(predictedScale.scale_geometric_px_per_m, 1)} px/m</dd></div><div><dt>FIELD visible</dt><dd>${numberText(predictedCoverage.visible_field_percent, 1)}%</dd></div><div><dt>Approx P95</dt><dd>${metricText(predictedApproximation.p95_px)}</dd></div></dl><p class="test-diagnostic-caption">Predicción transportada desde el HUMAN; no es ground truth del target.</p>`
      : "";
    return `${heading}<div class="test-camera-state-coverage-status is-${coverage.status || "unobservable"}">${String(coverage.status || "unobservable").toUpperCase()}</div><p>${statusCopy}</p><dl><div><dt>Target</dt><dd>Fotograma ${coverage.target_frame_index ?? "—"}</dd></div><div><dt>Modo</dt><dd>${coverage.coverage_mode || "—"}</dd></div><div><dt>Mejor HUMAN</dt><dd>Frame ${best.anchor_frame_index ?? coverage.best_anchor_id ?? "—"} · ${best.anchor_type === "assisted" ? "ASSISTED" : best.anchor_type === "direct" ? "DIRECT" : "—"}</dd></div><div><dt>Score / evidencia</dt><dd>${numberText(best.candidate_score)} / ${numberText(best.evidence_score)}</dd></div><div><dt>Distancia de estado</dt><dd>${numberText(best.state_distance)}</dd></div><div><dt>Transform recomendado</dt><dd>${cameraStateModelLabel(best.recommended_model)}</dd></div><div><dt>Inliers / ratio</dt><dd>${probe.consensus_inlier_count ?? "—"} / ${numberText(probe.consensus_inlier_ratio)}</dd></div><div><dt>Cobertura fuente / target</dt><dd>${numberText(probe.source_spatial_coverage)} / ${numberText(probe.target_spatial_coverage)}</dd></div><div><dt>View Transform P95</dt><dd>${metricText(probe.p95_px)}</dd></div><div><dt>Matches mutuos</dt><dd>${probe.mutual_match_count ?? "—"}</dd></div></dl>${predictedMarkup}${ranking ? `<h3>Top HUMAN candidatos</h3>${ranking}` : ""}${consensus}${failures}`;
  }
  function renderDerivedRefinement() {
    refineDerivedButton.disabled = !refinementReady();
    saveAssistedHumanButton.disabled = !assistedHumanReady();
    derivedRefinementModes.hidden = !derivedRefinement;
    derivedRefinementModes.querySelectorAll("[data-derived-refinement-model]").forEach((button) => {
      const name = button.dataset.derivedRefinementModel;
      const candidate = name === "propagated" ? viewTransformPropagation : derivedRefinement?.models?.[name];
      button.disabled = name !== "propagated" && Boolean(candidate?.error);
      button.classList.toggle("is-active", name === derivedRefinementModel);
      button.classList.toggle("is-recommended", name === derivedRefinement?.recommended_model);
    });
    derivedRefinementResult.hidden = !derivedRefinement;
    derivedRefinementResult.innerHTML = derivedRefinementMarkup(derivedRefinement);
  }
  function renderCameraStateExperiment() {
    const hasStates = humanStates.length > 0;
    const directStates = directHumanStates();
    const assistedCount = humanStates.length - directStates.length;
    pitchLengthInput.disabled = hasStates;
    pitchWidthInput.disabled = hasStates;
    cameraStateProfile.textContent = hasStates
      ? `Perfil FIELD bloqueado: ${numberText(cameraStateFieldProfile?.length, 1)} × ${numberText(cameraStateFieldProfile?.width, 1)} m · ${humanStates.length} HUMAN.${assistedCount ? ` ${assistedCount} ASSISTED no participa en análisis Camera State.` : ""}`
      : "Sin experimento HUMAN. Las dimensiones del campo siguen editables.";
    saveHumanButton.disabled = !frameLocked || !Number.isInteger(fixedFrame) || !anisotropicGeometryAvailable();
    evaluateCameraStateCoverageButton.disabled = !cameraStateCoverageReady() || cameraStateCoverageRunning;
    analyzeCameraStatesButton.disabled = directStates.length < 2;
    resetCameraStatesButton.disabled = !hasStates;
    cameraStateList.innerHTML = humanStates.map((state) => {
      const assisted = state.anchor_type === "assisted";
      const provenance = assisted ? `ASSISTED · desde frame ${state.provenance?.source_human_frame_index ?? "—"} · ${cameraStateModelLabel(state.provenance?.refinement_model)}` : "DIRECT · solve independiente";
      return `<article class="test-camera-state-card ${state.state_id === cameraStateReferenceId ? "is-reference" : ""}"><strong>CAMERA STATE · ${assisted ? "ASSISTED" : "DIRECT"}</strong><small>${stateFrameText(state)} · ${projectionModelLabel(state.projection_model)}<br>${provenance}</small>${humanDescriptorMarkup(state)}<div><button class="button button-secondary" type="button" data-camera-state-reference="${state.state_id}" ${state.state_id === cameraStateReferenceId ? "disabled" : ""}>Usar como referencia</button><button class="button button-secondary" type="button" data-camera-state-remove="${state.state_id}">Eliminar</button></div></article>`;
    }).join("") || "<p>Guarda solves anisotrópicos de fotogramas fijos para iniciar el experimento.</p>";
    cameraStateList.querySelectorAll("[data-camera-state-reference]").forEach((button) => button.addEventListener("click", () => setCameraStateReference(button.dataset.cameraStateReference)));
    cameraStateList.querySelectorAll("[data-camera-state-remove]").forEach((button) => button.addEventListener("click", () => removeHumanState(button.dataset.cameraStateRemove)));
    humanAtlasElement.hidden = !hasStates;
    humanAtlasElement.innerHTML = humanAtlasMarkup();
    cameraStateCoverageElement.hidden = !hasStates;
    cameraStateCoverageElement.innerHTML = cameraStateCoverageMarkup();
    cameraStateAnalysisElement.hidden = !cameraStateAnalysis;
    cameraStateAnalysisElement.innerHTML = analysisMarkup(cameraStateAnalysis);
    renderViewTransform();
    renderDerivedRefinement();
  }

  function resetSolution() {
    projected = null; cameraModelProbe = null; currentSolve = null; projectionVisible = true; cameraModel = "pure_homography"; diagnosticFocus = null; projectionToggle.hidden = !viewTransformPropagation; projectionToggle.textContent = "Ocultar proyección"; cameraModelModes.hidden = true; resultSection.hidden = true; result.textContent = "";
  }
  function resetAnnotations() {
    lineAnnotations.clear(); pointAnnotations.clear(); history.length = 0; draft = null; clearDerivedRefinement(); resetSolution();
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
  function selectedDerivedGeometry() {
    return selectedTransportedGeometry();
  }
  function selectedDerivedRefinedGeometry() {
    return selectedRefinedGeometry();
  }
  function draw() {
    if (!frameLocked || !canvas.width || !canvas.height) return;
    context.clearRect(0, 0, canvas.width, canvas.height); context.lineJoin = "round"; context.lineCap = "round";
    if ((projected || selectedDerivedRefinedGeometry() || selectedDerivedGeometry()) && projectionVisible) {
      drawProjectedGeometry();
    }
    const scale = Math.max(2, canvas.width / 900); context.strokeStyle = "#f2c94c"; context.fillStyle = "#f2c94c";
    lineAnnotations.forEach((annotation, name) => { const emphasis = (mode === "line" && name === selectedName) || (diagnosticFocus?.kind === "line" && diagnosticFocus.name === name); context.lineWidth = scale * (emphasis ? 1.75 : 1); context.beginPath(); context.moveTo(...annotation.points[0]); context.lineTo(...annotation.points[1]); context.stroke(); annotation.points.forEach(([x, y]) => { context.beginPath(); context.arc(x, y, context.lineWidth * 1.7, 0, Math.PI * 2); context.fill(); }); });
    pointAnnotations.forEach((annotation, name) => { const [x, y] = annotation.point; const emphasis = (mode === "point" && name === selectedName) || (diagnosticFocus?.kind === "point" && diagnosticFocus.name === name); context.beginPath(); context.arc(x, y, scale * (emphasis ? 2.8 : 1.9), 0, Math.PI * 2); context.fill(); });
    if (draft?.points.length) { const [x, y] = draft.points[0]; context.beginPath(); context.arc(x, y, scale * 2.4, 0, Math.PI * 2); context.fill(); }
  }
  function drawProjectedGeometry() {
    context.lineWidth = Math.max(1.5, canvas.width / 1150); context.strokeStyle = "rgba(103, 216, 255, .92)";
    const geometryToDraw = projected
      ? (cameraModelProbe?.models?.[cameraModel]?.projected_geometry || projected)
      : (selectedDerivedRefinedGeometry() || selectedDerivedGeometry());
    if (!geometryToDraw) return;
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
    invalidateCameraStateCoverageTarget();
    frameLocked = false; fixedTimestamp = null; fixedFrame = null; selectedName = null; clearViewTransformPropagation(); resetAnnotations(); clearOverlay(); frameInfo.textContent = "Navega por el vídeo y fija un fotograma para anotar."; updateUI();
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
    if (!selected) { invalidateCameraStateCoverageTarget(); testVideo.removeAttribute("src"); testVideo.load(); setEmpty("Abre un vídeo desde Biblioteca para comenzar."); return; }
    const sourceChanged = testVideo.dataset.assetId !== selected.assetId;
    if (sourceChanged) invalidateCameraStateCoverageTarget();
    if (sourceChanged && cameraStateVideoAssetId && cameraStateVideoAssetId !== selected.assetId) resetCameraStateExperiment();
    pendingStartTime = sourceVideo?.currentTime || 0;
    if (sourceChanged) { unlockFrame(); testVideo.dataset.assetId = selected.assetId; testVideo.src = selected.playbackUrl; testVideo.load(); setEmpty("Cargando vídeo…"); }
    else if (testVideo.readyState >= HTMLMediaElement.HAVE_METADATA) { if (!frameLocked && pendingStartTime !== null) { testVideo.currentTime = pendingStartTime; pendingStartTime = null; } empty.hidden = true; updatePlayback(); updateUI(); }
  }
  canvas.addEventListener("click", (event) => {
    if (!frameLocked || !selectedName) return;
    const rect = canvas.getBoundingClientRect(); const point = [(event.clientX - rect.left) * canvas.width / rect.width, (event.clientY - rect.top) * canvas.height / rect.height];
    clearDerivedRefinement();
    if (mode === "point") { const before = cloneAnnotation("point", pointAnnotations.get(selectedName)); const after = { point_name: selectedName, point }; applyAnnotation("point", selectedName, after); history.push({ kind: "point", name: selectedName, before, after: cloneAnnotation("point", after) }); resetSolution(); }
    else { if (!draft) draft = { name: selectedName, points: [] }; draft.points.push(point); if (draft.points.length === 2) { const before = cloneAnnotation("line", lineAnnotations.get(selectedName)); const after = { line_name: selectedName, points: draft.points }; applyAnnotation("line", selectedName, after); history.push({ kind: "line", name: selectedName, before, after: cloneAnnotation("line", after) }); draft = null; resetSolution(); } }
    updateUI(); draw();
  });
  lineModeButton.addEventListener("click", () => setMode("line")); pointModeButton.addEventListener("click", () => setMode("point"));
  document.querySelector("#test-undo").addEventListener("click", () => { clearDerivedRefinement(); if (draft?.points.length) { draft = null; updateUI(); draw(); return; } const operation = history.pop(); if (!operation) return; applyAnnotation(operation.kind, operation.name, operation.before); resetSolution(); updateUI(); draw(); });
  document.querySelector("#test-remove").addEventListener("click", () => { const target = annotationMap(mode); if (!selectedName || !target.has(selectedName)) return; clearDerivedRefinement(); const before = cloneAnnotation(mode, target.get(selectedName)); applyAnnotation(mode, selectedName, null); history.push({ kind: mode, name: selectedName, before, after: null }); draft = null; resetSolution(); updateUI(); draw(); });
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
  propagateHumanButton.addEventListener("click", async () => {
    const selected = window.SupervisorWorkspace?.getSelectedVideo();
    const reference = referenceHuman();
    const sourceGeometry = referenceHumanGeometry();
    if (!propagationReady() || !selected || !reference || !sourceGeometry) return;
    propagateHumanButton.disabled = true;
    viewTransformResult.hidden = false;
    viewTransformResult.textContent = "Transportando proyección desde HUMAN…";
    try {
      viewTransformPropagation = await api("/api/field/view-transform/probe", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          asset_id: selected.assetId,
          source_frame_index: reference.frame_index,
          target_frame_index: fixedFrame,
          source_projected_geometry: sourceGeometry,
        }),
      });
      clearDerivedRefinement();
      viewTransformModel = viewTransformPropagation.recommended_model;
      projectionVisible = true;
      projectionToggle.hidden = false;
      projectionToggle.textContent = "Ocultar proyección";
      renderViewTransform();
      renderDerivedRefinement();
      draw();
    } catch (error) {
      viewTransformResult.hidden = false;
      viewTransformResult.textContent = `No se pudo propagar: ${error.message}`;
      propagateHumanButton.disabled = false;
    }
  });
  viewTransformModes.querySelectorAll("[data-view-transform-model]").forEach((button) => button.addEventListener("click", () => {
    if (!viewTransformPropagation?.models?.[button.dataset.viewTransformModel]) return;
    viewTransformModel = button.dataset.viewTransformModel;
    clearDerivedRefinement();
    renderViewTransform();
    renderDerivedRefinement();
    draw();
  }));
  refineDerivedButton.addEventListener("click", async () => {
    const transportedGeometry = selectedTransportedGeometry();
    if (!refinementReady() || !transportedGeometry) return;
    const payload = {
      transported_geometry: transportedGeometry,
      video_width: canvas.width,
      video_height: canvas.height,
      point_annotations: [...pointAnnotations.values()],
      annotations: [...lineAnnotations.values()],
    };
    refineDerivedButton.disabled = true;
    derivedRefinementResult.hidden = false;
    derivedRefinementResult.textContent = "Refinando proyección DERIVED…";
    try {
      derivedRefinement = await api("/api/field/derived/refine", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      derivedRefinementModel = derivedRefinement.recommended_model;
      renderDerivedRefinement();
      draw();
    } catch (error) {
      derivedRefinementResult.hidden = false;
      derivedRefinementResult.textContent = `No se pudo refinar: ${error.message}`;
      refineDerivedButton.disabled = false;
    }
  });
  derivedRefinementModes.querySelectorAll("[data-derived-refinement-model]").forEach((button) => button.addEventListener("click", () => {
    const name = button.dataset.derivedRefinementModel;
    if (name !== "propagated" && derivedRefinement?.models?.[name]?.error) return;
    derivedRefinementModel = name;
    renderDerivedRefinement();
    draw();
  }));
  saveHumanButton.addEventListener("click", saveCurrentHumanState);
  saveAssistedHumanButton.addEventListener("click", saveAssistedHumanState);
  resetCameraStatesButton.addEventListener("click", resetCameraStateExperiment);
  evaluateCameraStateCoverageButton.addEventListener("click", async () => {
    const selected = window.SupervisorWorkspace?.getSelectedVideo();
    const anchors = cameraStateCoverageAnchors();
    if (!cameraStateCoverageReady() || !selected || !anchors.length) return;
    const requestId = ++cameraStateCoverageRequestId;
    cameraStateCoverageTarget = null;
    cameraStateCoverageError = null;
    cameraStateCoverageRunning = true;
    renderCameraStateExperiment();
    try {
      const coverage = await api("/api/field/camera-state/coverage/target", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          asset_id: selected.assetId,
          target_frame_index: fixedFrame,
          target_timestamp_seconds: fixedTimestamp,
          anchors: anchors.map((state) => ({
            anchor_id: state.state_id,
            asset_id: selected.assetId,
            anchor_frame_index: state.frame_index,
            anchor_type: state.anchor_type,
            projected_geometry: cloneJson(humanGeometry(state)),
            field_profile: cloneJson(state.field_profile),
            anchor_descriptor: state.view_descriptor ? cloneJson(state.view_descriptor) : null,
          })),
        }),
      });
      if (requestId !== cameraStateCoverageRequestId) return;
      cameraStateCoverageTarget = coverage;
      cameraStateCoverageRunning = false;
    } catch (error) {
      if (requestId !== cameraStateCoverageRequestId) return;
      cameraStateCoverageError = error.message;
      cameraStateCoverageRunning = false;
    }
    renderCameraStateExperiment();
  });
  analyzeCameraStatesButton.addEventListener("click", async () => {
    const states = directHumanStates();
    const referenceStateId = states.some((state) => state.state_id === cameraStateReferenceId)
      ? cameraStateReferenceId
      : states[0]?.state_id;
    if (states.length < 2 || !referenceStateId) return;
    analyzeCameraStatesButton.disabled = true;
    cameraStateAnalysisElement.hidden = false;
    cameraStateAnalysisElement.textContent = "Analizando relaciones entre estados HUMAN…";
    try {
      cameraStateAnalysis = await api("/api/field/camera-state/probe", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          projection_model: "anisotropic_2p",
          reference_state_id: referenceStateId,
          states,
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
