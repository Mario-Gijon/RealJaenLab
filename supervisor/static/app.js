const modeTabs = document.querySelectorAll(".mode-tab");
const modeLabel = document.querySelector("#mode-label");
const saveButton = document.querySelector("#save-button");
const undoButton = document.querySelector("#undo-button");
const saveStatus = document.querySelector("#save-status");
const saveStatusLabel = document.querySelector("#save-status-label");
const video = document.querySelector("#video");
const workspace = document.querySelector(".workspace");
const workArea = document.querySelector(".work-area");
const overlay = document.querySelector("#video-overlay");
const overlayContext = overlay.getContext("2d");
const videoSelect = document.querySelector("#video-select");
const workspaceMessage = document.querySelector("#workspace-message");
const workspaceTitle = document.querySelector("#workspace-title");
const workspaceDescription = document.querySelector("#workspace-description");
const playPauseButton = document.querySelector("#play-pause");
const previousFrameButtons = document.querySelectorAll("#previous-frame, #previous-frame-label");
const nextFrameButtons = document.querySelectorAll("#next-frame, #next-frame-label");
const timeline = document.querySelector("#timeline");
const currentTimeLabel = document.querySelector("#current-time");
const totalTimeLabel = document.querySelector("#total-time");
const frameReadout = document.querySelector("#frame-readout");
const debugToggle = document.querySelector("#debug-toggle");
const fieldToolButtons = document.querySelectorAll("[data-field-tool]");
const deleteFieldItemButton = document.querySelector("#delete-field-item");
const undoFieldEditButton = document.querySelector("#undo-field-edit");
const guidedWorkflowPanel = document.querySelector("#guided-workflow-panel");
const guidedWorkflowContent = document.querySelector("#guided-workflow-content");
const scanVideoButton = document.querySelector("#scan-video");
const beginGuidedCalibrationButton = document.querySelector("#begin-guided-calibration");
const openManualCalibrationButton = document.querySelector("#open-manual-calibration");
const guidedConfirmContinueButton = document.querySelector("#guided-confirm-continue");
const guidedFrameProgress = document.querySelector("#guided-frame-progress");
const strongAnchorCount = document.querySelector("#strong-anchor-count");
const cameraAtlasState = document.querySelector("#camera-atlas-state");
const fieldFrameProvenance = document.querySelector("#field-frame-provenance");
const fieldAtlasDebug = document.querySelector("#field-atlas-debug");
const atlasDimension95 = document.querySelector("#atlas-dimension-95");
const atlasDimension99 = document.querySelector("#atlas-dimension-99");
const atlasLoo = document.querySelector("#atlas-loo");
const atlasVisiblePitch = document.querySelector("#atlas-visible-pitch");
const atlasTimeCamera = document.querySelector("#atlas-time-camera");
const atlasFarNeighbor = document.querySelector("#atlas-far-neighbor");
const atlasGraph = document.querySelector("#atlas-graph");
const atlasLocalReconstruction = document.querySelector("#atlas-local-reconstruction");
const atlasCoverage = document.querySelector("#atlas-coverage");
const atlasRecommendedAnchors = document.querySelector("#atlas-recommended-anchors");
const suggestManualAnchorsButton = document.querySelector("#suggest-manual-anchors");
const fieldAtlasSelection = document.querySelector("#field-atlas-selection");
const guidedContextButton = document.querySelector("#guided-context");
const guidedSkipButton = document.querySelector("#guided-skip");
const calibrationPanel = document.querySelector("#calibration-panel");
const calibrationPanelHeader = document.querySelector("#calibration-panel-header");
const calibrationPanelToggle = document.querySelector("#calibration-panel-toggle");
const calibrationHeading = document.querySelector("#calibration-heading");
const calibrationLandmark = document.querySelector("#calibration-landmark");
const calibrationPitch = document.querySelector("#calibration-pitch");
const calibrationPitchGeometry = document.querySelector("#calibration-pitch-geometry");
const calibrationPitchMarkers = document.querySelector("#calibration-pitch-markers");
const calibrationPitchLines = document.querySelector("#calibration-pitch-lines");
const calibrationPitchArcs = document.querySelector("#calibration-pitch-arcs");
const pitchDiagramTopLabel = document.querySelector("#pitch-diagram-top-label");
const pitchDiagramLeftLabel = document.querySelector("#pitch-diagram-left-label");
const pitchDiagramRightLabel = document.querySelector("#pitch-diagram-right-label");
const calibrationStatus = document.querySelector("#calibration-status");
const calibrationSolverLoading = document.querySelector("#calibration-solver-loading");
const calibrationHint = document.querySelector("#calibration-hint");
const currentReferenceLabel = document.querySelector("#field-current-reference");
const calibrationConstraintType = document.querySelector("#calibration-constraint-type");
const removeCalibrationPointButton = document.querySelector("#remove-calibration-point");
const clearCalibrationButton = document.querySelector("#clear-calibration");
const exitCalibrationButton = document.querySelector("#exit-calibration");

let selectedVideo = null;
let debugOverlayEnabled = false;
let pointerVideoCoordinates = null;
let pointerBrowserPosition = null;
let displayedVideoRect = null;
let activeMode = "video";
let activeFieldTool = "select";
let fieldAnnotations = [];
let fieldCalibrations = [];
let fieldPropagations = [];
let fieldRefinedSegments = [];
let fieldPitchModel = { length_m: PITCH_DEFAULTS.lengthM, width_m: PITCH_DEFAULTS.widthM, status: "default" };
let fieldUndoStack = [];
let selectedFieldItem = null;
let fieldDragState = null;
let lineDraft = null;
let assistedDraft = null;
let annotationSequence = 0;
let reviewIsDirty = false;
let videoLoadSequence = 0;
let calibrationMode = false;
let calibrationDragState = null;
let calibrationSolveSequence = 0;
let calibrationSolvePending = null;
let calibrationSolveError = null;
let calibrationType = "point";
let calibrationSemantic = PITCH_MODEL.landmarks[0]?.id || null;
const selectedSemanticByType = { point: calibrationSemantic, line: null, arc: null };
let calibrationLineDraft = null;
let calibrationArcDraft = null;
let selectedCalibrationConstraint = null;
let calibrationPanelPosition = { x: 14, y: 60 };
let calibrationPanelDrag = null;
let cameraAtlasResult = null;
let cameraAtlasRequestSequence = 0;
let anchorSelectionResult = null;
let anchorSelectionInProgress = false;
let anchorSelectionError = null;
let viewScanWorkflow = null;
let fieldPitchSetup = null;

workArea.classList.remove("field-editing");
debugToggle.hidden = true;
undoButton.hidden = true;
saveStatus.hidden = true;
saveButton.hidden = true;

modeTabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    modeTabs.forEach((item) => item.classList.toggle("is-active", item === tab));
    activeMode = tab.dataset.mode;
    modeLabel.textContent = activeMode === "guided" ? "CALIBRACIÓN" : "MODO VÍDEO";
    const guided = activeMode === "guided";
    debugToggle.hidden = !guided;
    undoButton.hidden = !guided;
    saveStatus.hidden = !guided;
    saveButton.hidden = !guided;
    workArea.classList.toggle("field-editing", activeMode === "guided" && calibrationMode);
    selectedFieldItem = null;
    lineDraft = null;
    assistedDraft = null;
    setCalibrationMode(false);
    calibrationPanel.hidden = true;
    if (activeMode !== "guided") {
      setCalibrationMode(false);
      calibrationPanel.hidden = true;
    }
    updatePitchSetupWorkflow();
    drawOverlay();
  });
});

function setReviewDirty(isDirty) {
  reviewIsDirty = isDirty;
  saveStatus.classList.toggle("is-saved", !isDirty);
  saveStatusLabel.textContent = isDirty ? "SIN GUARDAR" : "GUARDADO";
}

function setCalibrationMode(enabled) {
  calibrationMode = Boolean(enabled);
  workspace.classList.toggle("is-calibrating", calibrationMode);
  workArea.classList.toggle("field-editing", activeMode === "guided" && calibrationMode);
}

function normalisePitchModel(value) {
  const length = Number(value?.length_m);
  const width = Number(value?.width_m);
  const hasValidLength = Number.isFinite(length) && length > PITCH_DEFAULTS.penaltyAreaDepthM * 2;
  const hasValidWidth = Number.isFinite(width)
    && width > (PITCH_DEFAULTS.goalHalfWidthM + PITCH_DEFAULTS.penaltyAreaDepthM) * 2;
  const status = hasValidLength && hasValidWidth && ["default", "estimated", "confirmed"].includes(value?.status)
    ? value.status
    : "default";
  return {
    length_m: hasValidLength ? length : PITCH_DEFAULTS.lengthM,
    width_m: hasValidWidth ? width : PITCH_DEFAULTS.widthM,
    status,
  };
}

function formatPitchDimension(value) {
  return Number.isInteger(value) ? String(value) : String(Number(value.toFixed(2)));
}

function applyPitchModel(value) {
  fieldPitchModel = normalisePitchModel(value);
  PITCH_MODEL = buildPitchModel(fieldPitchModel.length_m, fieldPitchModel.width_m);
  if (!semanticDefinitionsForType(calibrationType).some((item) => item.id === calibrationSemantic)) {
    calibrationSemantic = PITCH_MODEL.landmarks[0]?.id || null;
  }
  if (calibrationHeading) {
    calibrationHeading.textContent = `REFERENCIA ${formatPitchDimension(PITCH_MODEL.lengthM)} × ${formatPitchDimension(PITCH_MODEL.widthM)} M`;
  }
}

function clampCalibrationPanelPosition(position = calibrationPanelPosition) {
  const maximumX = Math.max(0, workArea.clientWidth - calibrationPanel.offsetWidth);
  const maximumY = Math.max(0, workArea.clientHeight - calibrationPanel.offsetHeight);
  return {
    x: Math.min(Math.max(position.x, 0), maximumX),
    y: Math.min(Math.max(position.y, 0), maximumY),
  };
}

function positionCalibrationPanel(position = calibrationPanelPosition) {
  calibrationPanelPosition = clampCalibrationPanelPosition(position);
  calibrationPanel.style.left = `${calibrationPanelPosition.x}px`;
  calibrationPanel.style.top = `${calibrationPanelPosition.y}px`;
}

function clampVisibleCalibrationPanel() {
  if (!calibrationPanel.hidden) positionCalibrationPanel();
}

// These helpers are the shared mapping boundary for future FIELD and PEOPLE overlays.
// They use the actual contained video image, rather than the larger workspace/video element.
function videoToDisplayCoordinates({ x, y }) {
  if (!displayedVideoRect || !video.videoWidth || !video.videoHeight) return null;
  return {
    x: displayedVideoRect.left + (x / video.videoWidth) * displayedVideoRect.width,
    y: displayedVideoRect.top + (y / video.videoHeight) * displayedVideoRect.height,
  };
}

function browserToVideoCoordinates({ clientX, clientY }) {
  if (!displayedVideoRect || !video.videoWidth || !video.videoHeight) return null;
  const { left, top, width, height } = displayedVideoRect;
  if (clientX < left || clientX > left + width || clientY < top || clientY > top + height) return null;
  return {
    x: ((clientX - left) / width) * video.videoWidth,
    y: ((clientY - top) / height) * video.videoHeight,
  };
}

function getDisplayedVideoRect() {
  if (!video.videoWidth || !video.videoHeight) return null;
  const videoBounds = video.getBoundingClientRect();
  const videoAspectRatio = video.videoWidth / video.videoHeight;
  const elementAspectRatio = videoBounds.width / videoBounds.height;
  const width = elementAspectRatio > videoAspectRatio
    ? videoBounds.height * videoAspectRatio
    : videoBounds.width;
  const height = elementAspectRatio > videoAspectRatio
    ? videoBounds.height
    : videoBounds.width / videoAspectRatio;
  return {
    left: videoBounds.left + (videoBounds.width - width) / 2,
    top: videoBounds.top + (videoBounds.height - height) / 2,
    width,
    height,
  };
}

function currentFrameNumber() {
  return selectedVideo?.fps ? Math.floor(video.currentTime * selectedVideo.fps) : 0;
}

function annotationsForCurrentFrame() {
  return fieldAnnotations.filter((annotation) => (
    annotation.video === selectedVideo?.name && annotation.frame === currentFrameNumber()
  ));
}

function calibrationForCurrentFrame() {
  return fieldCalibrations.find((calibration) => (
    calibration.video === selectedVideo?.name && calibration.frame === currentFrameNumber()
  ));
}

function propagatedFrameForCurrentFrame() {
  const frame = currentFrameNumber();
  for (const propagation of fieldPropagations) {
    if (propagation.video !== selectedVideo?.name) continue;
    const propagatedFrame = propagation.frames?.find((item) => item.frame === frame);
    if (propagatedFrame?.homography) return propagatedFrame;
  }
  return null;
}

function refinedFrameForCurrentFrame() {
  const frame = currentFrameNumber();
  const candidates = fieldRefinedSegments
    .filter((segment) => (
      segment.video === selectedVideo?.name
      && segment.leftAnchorFrame <= frame
      && segment.rightAnchorFrame >= frame
    ))
    .map((segment) => ({
      segment,
      frame: segment.frames?.find((item) => item.frame === frame),
    }))
    .filter((candidate) => Array.isArray(candidate.frame?.homography));
  candidates.sort((first, second) => (
    (first.segment.rightAnchorFrame - first.segment.leftAnchorFrame)
    - (second.segment.rightAnchorFrame - second.segment.leftAnchorFrame)
  ));
  return candidates[0]?.frame || null;
}

function existingProjectionForCurrentFrame() {
  const refinedFrame = refinedFrameForCurrentFrame();
  if (Array.isArray(refinedFrame?.homography)) {
    return { homography: refinedFrame.homography, source: "refined" };
  }
  const propagatedFrame = propagatedFrameForCurrentFrame();
  if (Array.isArray(propagatedFrame?.homography)) {
    return { homography: propagatedFrame.homography, source: "propagated" };
  }
  return null;
}

function hasInitialHomography(calibration) {
  return Array.isArray(calibration?.initialHomography) && calibration.initialHomography.length === 3;
}

function hasSolvedManualCalibration(calibration = calibrationForCurrentFrame()) {
  return calibration?.source === "manual"
    && Array.isArray(calibration.homography)
    && calibration.homography.length === 3;
}

function fieldHomographyForCurrentFrame() {
  const calibration = calibrationForCurrentFrame();
  if (hasSolvedManualCalibration(calibration)) return calibration.homography;
  return refinedFrameForCurrentFrame()?.homography || propagatedFrameForCurrentFrame()?.homography || null;
}

function currentCameraAtlas() {
  return cameraAtlasResult?.video === selectedVideo?.name ? cameraAtlasResult.result : null;
}

function currentAtlasAnchor() {
  const frame = currentFrameNumber();
  return currentCameraAtlas()?.atlas?.anchors?.find((anchor) => anchor.frame === frame) || null;
}

function updateFieldStatus() {
  const atlas = currentCameraAtlas();
  const strongAnchors = Number(atlas?.summary?.strong_anchor_count || 0);
  strongAnchorCount.textContent = String(strongAnchors);
  cameraAtlasState.textContent = atlas
    ? `Estado: ${strongAnchors < 2 ? "Datos insuficientes" : "Analizado"}`
    : "Estado: Pendiente";
  fieldAtlasDebug.hidden = !debugOverlayEnabled || !atlas;
  if (atlas) {
    const summary = atlas.summary || {};
    atlasDimension95.textContent = `Dimensión 95 %: ${summary.dimension_95 ?? "—"}`;
    atlasDimension99.textContent = `Dimensión 99 %: ${summary.dimension_99 ?? "—"}`;
    const loo = summary.leave_one_out?.median_of_median_error_px;
    atlasLoo.textContent = Number.isFinite(loo) ? `LOO mediano: ${loo.toFixed(1)} px` : "LOO: pendiente";
    const structure = atlas.structure_summary || {};
    const timeCamera = structure.time_camera_spearman;
    const farNeighbor = structure.nearest_far_time_median_px;
    const reconstruction = structure.local_reconstruction?.median_of_median_error_px;
    atlasTimeCamera.textContent = Number.isFinite(timeCamera) ? `Temporal ↔ cámara ρ: ${timeCamera.toFixed(2)}` : "Temporal ↔ cámara ρ: —";
    atlasFarNeighbor.textContent = Number.isFinite(farNeighbor) ? `Vecino lejano: ${farNeighbor.toFixed(1)} px` : "Vecino lejano: —";
    atlasGraph.textContent = Number.isInteger(structure.graph_k) && Number.isInteger(structure.graph_components)
      ? `Grafo: k=${structure.graph_k} · ${structure.graph_components} componente(s)`
      : "Grafo: —";
    atlasLocalReconstruction.textContent = Number.isFinite(reconstruction)
      ? `Reconstrucción local: ${reconstruction.toFixed(1)} px`
      : "Reconstrucción local: —";
    const coverage = atlas.coverage_summary || {};
    const probeCount = Number(coverage.probe_count || 0);
    if (probeCount > 0 && Number.isFinite(coverage.coverage_median_px)) {
      atlasCoverage.textContent = `Cobertura probes · Mediana: ${coverage.coverage_median_px.toFixed(1)} px · P90: ${coverage.coverage_p90_px.toFixed(1)} px · Máximo: ${coverage.coverage_max_px.toFixed(1)} px`;
    } else {
      atlasCoverage.textContent = "Cobertura: sin probes";
    }
    const recommendedFrames = Array.isArray(coverage.recommended_frames) ? coverage.recommended_frames : [];
    atlasRecommendedAnchors.hidden = recommendedFrames.length === 0;
    atlasRecommendedAnchors.textContent = recommendedFrames.length
      ? `Nuevos anchors sugeridos: ${recommendedFrames.join(", ")}`
      : "";
  }
  const calibration = calibrationForCurrentFrame();
  const atlasAnchor = currentAtlasAnchor();
  atlasVisiblePitch.hidden = true;
  if (atlasAnchor) {
    fieldFrameProvenance.textContent = "ANCHOR FUERTE · HUMANO";
    const fraction = Number(atlasAnchor.visibility?.visible_fraction);
    if (debugOverlayEnabled && Number.isFinite(fraction)) {
      atlasVisiblePitch.textContent = `Campo visible: ${(fraction * 100).toFixed(0)} %`;
      atlasVisiblePitch.hidden = false;
    }
    return;
  }
  if (hasSolvedManualCalibration(calibration)) {
    fieldFrameProvenance.textContent = "CALIBRACIÓN MANUAL · APOYO";
    return;
  }
  if (Array.isArray(refinedFrameForCurrentFrame()?.homography) || Array.isArray(propagatedFrameForCurrentFrame()?.homography)) {
    fieldFrameProvenance.textContent = "APOYO AUTOMÁTICO · NO VALIDADO";
    return;
  }
  fieldFrameProvenance.textContent = "SIN ESTADO FIELD";
}

function appendAnchorSelectionLine(fragment, text, emphasis = false) {
  const line = document.createElement(emphasis ? "strong" : "span");
  line.textContent = text;
  fragment.append(line);
}

function updateAnchorSelectorDebug() {
  const visible = debugOverlayEnabled && activeMode === "field";
  suggestManualAnchorsButton.hidden = !visible;
  suggestManualAnchorsButton.disabled = !visible || !selectedVideo || anchorSelectionInProgress;
  suggestManualAnchorsButton.textContent = anchorSelectionInProgress ? "Analizando vistas…" : "Sugerir anchors";

  const selection = anchorSelectionResult?.video === selectedVideo?.name
    ? anchorSelectionResult.result?.selection
    : null;
  fieldAtlasSelection.hidden = !visible || !selection;
  if (!visible || !selection) return;

  const fragment = document.createDocumentFragment();
  const recommendations = Array.isArray(selection.recommendations) ? selection.recommendations : [];
  if (recommendations.length) {
    appendAnchorSelectionLine(fragment, "ANCHORS SUGERIDOS", true);
    recommendations.forEach((item) => {
      const nearest = item.nearest_anchor || {};
      const view = item.view || {};
      const visual = item.visual_diagnostics || {};
      const visiblePercent = Number(view.visible_fraction);
      const qualityPercent = Number(item.quality_score);
      appendAnchorSelectionLine(
        fragment,
        `#${item.rank} · f ${item.frame} · Cobertura: ${Number(item.coverage_distance_px).toFixed(1)} px · Calidad visual: ${Number.isFinite(qualityPercent) ? (qualityPercent * 100).toFixed(0) : "—"} % · Vista: ${view.side || "desconocida"} · Zoom: ${view.zoom || "desconocido"} · Campo visible: ${Number.isFinite(visiblePercent) ? (visiblePercent * 100).toFixed(0) : "—"} % · Líneas visibles: ${view.visible_lines ?? 0}${nearest.frame !== undefined ? ` · Anchor HUMAN: ${nearest.frame} (${Number(nearest.median_px).toFixed(1)} px)` : ""}`,
      );
      const semantics = Array.isArray(view.visible_line_semantics) && view.visible_line_semantics.length
        ? `Líneas: ${view.visible_line_semantics.join(", ")}`
        : "Líneas: —";
      appendAnchorSelectionLine(fragment, `  ${semantics}`);
      appendAnchorSelectionLine(fragment, `  Verde: ${(Number(visual.green_fraction || 0) * 100).toFixed(1)} % · Línea: ${(Number(visual.line_like_fraction || 0) * 100).toFixed(2)} % · Nitidez: ${Number(visual.sharpness || 0).toFixed(2)}`);
    });
  }
  const reviews = Array.isArray(selection.isolated_anchor_reviews) ? selection.isolated_anchor_reviews : [];
  if (reviews.length) {
    appendAnchorSelectionLine(fragment, "ANCHORS HUMAN A REVISAR", true);
    reviews.forEach((item) => appendAnchorSelectionLine(
      fragment,
      `Frame ${item.frame} · HUMAN más cercano: ${item.nearest_anchor_frame} · ${Number(item.nearest_anchor_distance_px).toFixed(1)} px · revisar o confirmar vista única`,
    ));
  }
  fieldAtlasSelection.replaceChildren(fragment);
}

async function analyzeCameraAtlas() {
  if (!selectedVideo) return null;
  const imageWidth = video.videoWidth || selectedVideo.width;
  const imageHeight = video.videoHeight || selectedVideo.height;
  if (!imageWidth || !imageHeight) return null;

  const requestSequence = ++cameraAtlasRequestSequence;
  const videoName = selectedVideo.name;
  try {
    const response = await fetch("/api/field-camera-atlas/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        calibrations: fieldCalibrations.filter((calibration) => calibration.video === videoName),
        propagations: fieldPropagations.filter((propagation) => propagation.video === videoName),
        refined_segments: fieldRefinedSegments.filter((segment) => segment.video === videoName),
        image_width: imageWidth,
        image_height: imageHeight,
        video_name: videoName,
      }),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || "No se pudo analizar Camera Atlas");
    if (requestSequence !== cameraAtlasRequestSequence || selectedVideo?.name !== videoName) return;
    cameraAtlasResult = { video: videoName, result };
  } catch (error) {
    if (requestSequence !== cameraAtlasRequestSequence) return;
    cameraAtlasResult = null;
  }
  drawOverlay();
  return cameraAtlasResult?.video === videoName ? cameraAtlasResult.result : null;
}

async function requestAnchorSelection() {
  if (!selectedVideo) return { ok: false, error: "No hay un vídeo seleccionado." };
  if (anchorSelectionInProgress) return { ok: false, error: "El selector de anchors ya está analizando el vídeo." };
  const imageWidth = video.videoWidth || selectedVideo.width;
  const imageHeight = video.videoHeight || selectedVideo.height;
  if (!imageWidth || !imageHeight) return { ok: false, error: "No se pudo determinar la resolución del vídeo." };

  const videoName = selectedVideo.name;
  // A selector result belongs to a precise video/review state. Never let an
  // earlier result stand in for the analysis that is about to be requested.
  anchorSelectionResult = null;
  anchorSelectionError = null;
  anchorSelectionInProgress = true;
  updateAnchorSelectorDebug();
  try {
    const response = await fetch("/api/field-anchor-selector/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        filename: videoName,
        calibrations: fieldCalibrations.filter((calibration) => calibration.video === videoName),
        propagations: fieldPropagations.filter((propagation) => propagation.video === videoName),
        refined_segments: fieldRefinedSegments.filter((segment) => segment.video === videoName),
        image_width: imageWidth,
        image_height: imageHeight,
      }),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || "No se pudieron sugerir anchors");
    if (!result?.selection || !Array.isArray(result.selection.recommendations)) {
      throw new Error("El selector devolvió una respuesta no válida.");
    }
    if (selectedVideo?.name === videoName) {
      anchorSelectionResult = { video: videoName, result };
      return { ok: true, result };
    }
    return { ok: false, error: "El vídeo cambió mientras se analizaban los anchors." };
  } catch (error) {
    const message = error instanceof Error ? error.message : "No se pudo ejecutar el selector de anchors.";
    if (selectedVideo?.name === videoName) {
      anchorSelectionResult = null;
      anchorSelectionError = message;
    }
    return { ok: false, error: message };
  } finally {
    if (selectedVideo?.name === videoName) anchorSelectionInProgress = false;
    drawOverlay();
  }
}

async function suggestManualAnchors() {
  await requestAnchorSelection();
}

function seekToFrame(frame) {
  if (!selectedVideo?.fps || !Number.isFinite(frame)) return;
  video.pause();
  video.currentTime = Math.min(Math.max((frame + 0.01) / selectedVideo.fps, 0), Math.max(video.duration || 0, 0));
}

function pitchSetupMeasurementFrames() {
  return Array.isArray(fieldPitchSetup?.measurement_frames) ? fieldPitchSetup.measurement_frames : [];
}

function pitchSetupUsedFrames() {
  return Array.isArray(fieldPitchSetup?.used_frames) ? fieldPitchSetup.used_frames : [];
}

function pitchSetupCurrentFrame() {
  return pitchSetupMeasurementFrames().find((frame) => !pitchSetupUsedFrames().includes(frame)) ?? null;
}

function pitchSetupObservations() {
  const usedFrames = new Set(pitchSetupUsedFrames());
  return fieldCalibrations
    .filter((calibration) => (
      calibration.video === selectedVideo?.name
      && calibration.source === "manual"
      && usedFrames.has(Number(calibration.frame))
      && hasSolvedManualCalibration(calibration)
    ))
    .map((calibration) => ({
      frame: Number(calibration.frame),
      image_width: video.videoWidth,
      image_height: video.videoHeight,
      constraints: calibrationConstraints(calibration)
        .filter((constraint) => constraint.source === "manual" && constraint.semantic && constraint.image)
        .map((constraint) => ({
          type: constraint.type,
          semantic: constraint.semantic,
          image: JSON.parse(JSON.stringify(constraint.image)),
          source: "manual",
        })),
    }));
}

function pitchSetupMissingEvidence(result) {
  const missing = result?.needs_more_evidence || {};
  const labels = [];
  if (missing.length) labels.push("longitud");
  if (missing.width) labels.push("anchura");
  if (missing.usable_frames) labels.push("mediciones útiles");
  return labels;
}

function formatPitchInterval(interval) {
  if (!Array.isArray(interval) || interval.length !== 2) return "pendiente";
  return `${formatPitchDimension(Number(interval[0]))}–${formatPitchDimension(Number(interval[1]))} m`;
}

function updatePitchSetupWorkflow() {
  const active = activeMode === "guided";
  const state = fieldPitchSetup?.video === selectedVideo?.name ? fieldPitchSetup : null;
  const used = pitchSetupUsedFrames();
  const frames = pitchSetupMeasurementFrames();
  const currentFrame = pitchSetupCurrentFrame();
  const result = state?.result || null;
  const isConfirmed = fieldPitchModel.status === "confirmed";

  guidedWorkflowPanel.hidden = !active || calibrationMode;
  guidedFrameProgress.hidden = !(calibrationMode && state?.status === "measuring" && currentFrame !== null);
  guidedFrameProgress.textContent = calibrationMode && currentFrame !== null ? `TERRENO · ${used.length + 1} / ${frames.length}` : "";
  guidedConfirmContinueButton.hidden = !calibrationMode || state?.status !== "measuring" || currentFrame === null;
  guidedConfirmContinueButton.disabled = !(
    currentFrame === currentFrameNumber()
    && hasSolvedManualCalibration(calibrationForCurrentFrame())
    && !calibrationSolvePending
    && !state?.processing
  );

  if (!active || calibrationMode) return;

  const fragment = document.createDocumentFragment();
  const add = (text, strong = false, warning = false) => {
    const element = document.createElement(strong ? "strong" : "span");
    element.textContent = text;
    if (warning) element.classList.add("guided-warning");
    fragment.append(element);
  };
  add(`1 · TERRENO${isConfirmed ? "  ✓" : ""}`, true);
  if (!selectedVideo) {
    add("Selecciona un vídeo para comenzar.");
  } else if (isConfirmed && !state) {
    add(`${formatPitchDimension(fieldPitchModel.length_m)} × ${formatPitchDimension(fieldPitchModel.width_m)} m confirmado.`);
  } else if (!state) {
    add("Mide el terreno con vistas seleccionadas automáticamente del vídeo.");
  } else if (state.processing === "scan") {
    add("Analizando vídeo para seleccionar mediciones…");
  } else if (state.processing === "estimate") {
    add("Estimando las dimensiones del terreno…");
  } else if (state.error) {
    add(`No se pudo continuar: ${state.error}`, false, true);
  } else if (state.status === "measuring") {
    add(`${used.length + 1} / ${frames.length}`);
    add(`Siguiente medición: fotograma ${currentFrame}.`);
  } else if (state.status === "ready_to_confirm" && result?.pitch_model) {
    add("Terreno estimado", true);
    add(`${formatPitchDimension(result.pitch_model.length_m)} × ${formatPitchDimension(result.pitch_model.width_m)} m`);
    add(`Longitud: ${formatPitchInterval(result.confidence?.length_interval_m)} · Anchura: ${formatPitchInterval(result.confidence?.width_interval_m)}`);
  } else if (state.status === "insufficient") {
    add("No hay evidencia geométrica suficiente para estimar el terreno con confianza.", false, true);
    const missing = pitchSetupMissingEvidence(result);
    if (missing.length) add(`Falta confianza en: ${missing.join(", ")}.`);
  } else if (state.status === "confirmed") {
    add(`${formatPitchDimension(fieldPitchModel.length_m)} × ${formatPitchDimension(fieldPitchModel.width_m)} m confirmado.`);
  }
  add("2 · PARTIDO", true);
  add(isConfirmed ? "Disponible próximamente" : "Bloqueado hasta confirmar el terreno.");
  guidedWorkflowContent.replaceChildren(fragment);

  scanVideoButton.hidden = (Boolean(state) && state.status !== "error") || isConfirmed;
  scanVideoButton.disabled = !selectedVideo;
  beginGuidedCalibrationButton.hidden = state?.status !== "ready_to_confirm";
  beginGuidedCalibrationButton.disabled = Boolean(state?.processing);
  openManualCalibrationButton.hidden = state?.status !== "measuring" || currentFrame === null;
  openManualCalibrationButton.disabled = Boolean(state?.processing);
  guidedContextButton.hidden = true;
  guidedSkipButton.hidden = true;
}

function openPitchSetupMeasurement(frame) {
  if (!Number.isFinite(frame)) return;
  const open = () => enterCalibrationMode();
  if (currentFrameNumber() === frame) open();
  else {
    video.addEventListener("seeked", open, { once: true });
    seekToFrame(frame);
  }
}

async function startPitchSetup() {
  if (!selectedVideo || fieldPitchSetup?.processing) return;
  const videoName = selectedVideo.name;
  fieldPitchSetup = { video: videoName, status: "starting", processing: "scan", scan: null, measurement_frames: [], used_frames: [], result: null, error: null };
  updatePitchSetupWorkflow();
  try {
    const scanResponse = await fetch("/api/field-view-scan/analyze", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ filename: videoName }),
    });
    const scanPayload = await scanResponse.json();
    if (!scanResponse.ok || !scanPayload?.scan) throw new Error(scanPayload.detail || "No se pudo analizar el vídeo.");
    const selectResponse = await fetch("/api/field-pitch-setup/select", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ scan: scanPayload.scan }),
    });
    const selection = await selectResponse.json();
    if (!selectResponse.ok || !Array.isArray(selection.selected_frames) || !selection.selected_frames.length) {
      throw new Error(selection.detail || "No se encontraron vistas útiles para medir el terreno.");
    }
    if (selectedVideo?.name !== videoName) return;
    fieldPitchSetup = {
      video: videoName,
      status: "measuring",
      processing: null,
      scan: scanPayload.scan,
      measurement_frames: selection.selected_frames,
      used_frames: [],
      result: null,
      error: null,
    };
    setReviewDirty(true);
    await saveCurrentReview();
    openPitchSetupMeasurement(selection.selected_frames[0]);
  } catch (error) {
    if (selectedVideo?.name === videoName) {
      fieldPitchSetup = { video: videoName, status: "error", processing: null, scan: null, measurement_frames: [], used_frames: [], result: null, error: error instanceof Error ? error.message : "Error desconocido." };
    }
  }
  updatePitchSetupWorkflow();
}

async function completePitchSetupMeasurement() {
  const state = fieldPitchSetup;
  const frame = pitchSetupCurrentFrame();
  if (!state || state.status !== "measuring" || frame === null || frame !== currentFrameNumber()) return;
  if (!hasSolvedManualCalibration(calibrationForCurrentFrame()) || calibrationSolvePending) return;
  state.used_frames = [...pitchSetupUsedFrames(), frame];
  state.processing = "estimate";
  setReviewDirty(true);
  updatePitchSetupWorkflow();
  await saveCurrentReview();
  try {
    const response = await fetch("/api/field-pitch-setup/estimate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ observations: pitchSetupObservations() }),
    });
    const payload = await response.json();
    if (!response.ok || !payload?.result) throw new Error(payload.detail || "No se pudo estimar el terreno.");
    state.result = payload.result;
    if (payload.result.status === "ready_to_confirm") {
      state.status = "ready_to_confirm";
      state.processing = null;
      exitCalibrationMode();
    } else if (payload.result.status === "needs_more_evidence") {
      const nextResponse = await fetch("/api/field-pitch-setup/next-frame", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ scan: state.scan, used_frames: state.used_frames }),
      });
      const nextPayload = await nextResponse.json();
      if (!nextResponse.ok) throw new Error(nextPayload.detail || "No se pudo seleccionar otra medición.");
      const nextFrame = nextPayload.recommendation?.frame;
      if (Number.isFinite(nextFrame)) {
        state.measurement_frames = [...pitchSetupMeasurementFrames(), nextFrame];
        state.status = "measuring";
        state.processing = null;
        setReviewDirty(true);
        await saveCurrentReview();
        openPitchSetupMeasurement(nextFrame);
      } else {
        state.status = "insufficient";
        state.processing = null;
        exitCalibrationMode();
      }
    } else {
      throw new Error(payload.result.detail || "La estimación del terreno no pudo completarse.");
    }
  } catch (error) {
    state.status = "error";
    state.processing = null;
    state.error = error instanceof Error ? error.message : "Error desconocido.";
  }
  setReviewDirty(true);
  await saveCurrentReview();
  updatePitchSetupWorkflow();
}

async function confirmPitchSetup() {
  const result = fieldPitchSetup?.result;
  if (fieldPitchSetup?.status !== "ready_to_confirm" || !result?.pitch_model) return;
  applyPitchModel({ ...result.pitch_model, status: "confirmed" });
  fieldPitchSetup.status = "confirmed";
  fieldPitchSetup.processing = null;
  setReviewDirty(true);
  await saveCurrentReview();
  updatePitchSetupWorkflow();
  drawOverlay();
}

function currentViewScanRecommendation() {
  if (!viewScanWorkflow?.started) return null;
  return viewScanWorkflow.recommendations[viewScanWorkflow.index] || null;
}

function updateViewScanWorkflow() {
  const active = activeMode === "guided";
  const state = viewScanWorkflow;
  const recommendation = currentViewScanRecommendation();
  const progressVisible = Boolean(calibrationMode && recommendation);
  guidedFrameProgress.hidden = !progressVisible;
  guidedFrameProgress.textContent = progressVisible ? `Frame ${state.index + 1} / ${state.recommendations.length}` : "";
  const canConfirm = progressVisible
    && currentFrameNumber() === recommendation.frame
    && hasSolvedManualCalibration(calibrationForCurrentFrame())
    && !calibrationSolvePending;
  guidedConfirmContinueButton.hidden = !progressVisible;
  guidedConfirmContinueButton.disabled = !canConfirm;
  guidedWorkflowPanel.hidden = !active || calibrationMode;
  if (!active || calibrationMode) return;

  const fragment = document.createDocumentFragment();
  const add = (text, strong = false) => {
    const element = document.createElement(strong ? "strong" : "span");
    element.textContent = text;
    fragment.append(element);
  };
  if (!selectedVideo) {
    add("Selecciona un vídeo para iniciar la calibración guiada.", true);
  } else if (!state || state.video !== selectedVideo.name) {
    add("Analiza el vídeo para localizar vistas representativas del campo.", true);
  } else if (state.scanning) {
    add("Analizando geometría de cámara…", true);
  } else if (state.error) {
    add(`No se pudo analizar el vídeo: ${state.error}`, true);
  } else if (!state.started) {
    add(`Muestras analizadas: ${state.summary.sample_count} · Vistas detectadas: ${state.summary.recommendation_count}`, true);
  } else {
    const recommendation = currentViewScanRecommendation();
    if (recommendation) add(`Frame ${state.index + 1} de ${state.recommendations.length} · fotograma ${recommendation.frame}`, true);
    else {
      add("Calibración inicial completada", true);
      add(`Recomendaciones: ${state.recommendations.length} · Calibraciones HUMAN completadas: ${state.completedFrames.size} · Omitidos: ${state.skippedFrames.size}`);
    }
  }
  guidedWorkflowContent.replaceChildren(fragment);

  const hasScan = Boolean(state?.scan && !state.scanning && !state.error);
  scanVideoButton.hidden = hasScan;
  scanVideoButton.disabled = !selectedVideo || Boolean(state?.scanning);
  scanVideoButton.textContent = state?.scanning ? "Analizando geometría de cámara…" : "Analizar vídeo";
  beginGuidedCalibrationButton.hidden = !hasScan || Boolean(state?.started);
  openManualCalibrationButton.hidden = !recommendation;
  guidedContextButton.hidden = !recommendation;
  guidedSkipButton.hidden = !recommendation;
  openManualCalibrationButton.disabled = !recommendation;
  guidedContextButton.disabled = !recommendation;
  guidedSkipButton.disabled = !recommendation;

}

async function scanVideoViewStates() {
  if (!selectedVideo || viewScanWorkflow?.scanning) return;
  const videoName = selectedVideo.name;
  viewScanWorkflow = { video: videoName, scanning: true, error: null, scan: null, summary: null, started: false, recommendations: [], index: 0, completedFrames: new Set(), skippedFrames: new Set() };
  updatePitchSetupWorkflow();
  try {
    const response = await fetch("/api/field-view-scan/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ filename: videoName }),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || "El escáner no devolvió un resultado válido.");
    if (selectedVideo?.name !== videoName) return;
    if (!result?.scan || !Array.isArray(result.scan.recommendations)) throw new Error("La respuesta del escáner no contiene recomendaciones válidas.");
    viewScanWorkflow = {
      video: videoName,
      scanning: false,
      error: null,
      scan: result.scan,
      summary: result.summary || {},
      started: false,
      recommendations: [...result.scan.recommendations].sort((a, b) => a.frame - b.frame),
      index: 0,
      completedFrames: new Set(),
      skippedFrames: new Set(),
    };
  } catch (error) {
    if (selectedVideo?.name === videoName) {
      viewScanWorkflow = { video: videoName, scanning: false, error: error instanceof Error ? error.message : "Error desconocido.", scan: null, summary: null, started: false, recommendations: [], index: 0, completedFrames: new Set(), skippedFrames: new Set() };
    }
  }
  updatePitchSetupWorkflow();
}

function beginViewScanCalibration() {
  if (!viewScanWorkflow?.scan) return;
  viewScanWorkflow.started = true;
  viewScanWorkflow.index = 0;
  const recommendation = currentViewScanRecommendation();
  if (recommendation) seekToFrame(recommendation.frame);
  updatePitchSetupWorkflow();
}

function openManualCalibrationForView() {
  const recommendation = currentViewScanRecommendation();
  if (!recommendation) return;
  seekToFrame(recommendation.frame);
  enterCalibrationMode();
}

function playViewScanContext() {
  const recommendation = currentViewScanRecommendation();
  if (!recommendation || !selectedVideo?.fps) return;
  viewScanWorkflow.contextEndTime = Math.min((recommendation.frame / selectedVideo.fps) + 3, video.duration || Infinity);
  video.currentTime = Math.max(0, (recommendation.frame / selectedVideo.fps) - 3);
  video.play();
}

function skipViewScanRecommendation() {
  const recommendation = currentViewScanRecommendation();
  if (!recommendation) return;
  viewScanWorkflow.skippedFrames.add(recommendation.frame);
  viewScanWorkflow.index += 1;
  const next = currentViewScanRecommendation();
  if (next) seekToFrame(next.frame);
  updatePitchSetupWorkflow();
}

async function confirmAndContinueViewScan() {
  const recommendation = currentViewScanRecommendation();
  if (!recommendation || currentFrameNumber() !== recommendation.frame || !hasSolvedManualCalibration(calibrationForCurrentFrame()) || calibrationSolvePending) return;
  guidedConfirmContinueButton.disabled = true;
  const saved = await saveCurrentReview();
  if (!saved) {
    updateViewScanWorkflow();
    return;
  }
  viewScanWorkflow.completedFrames.add(recommendation.frame);
  viewScanWorkflow.index += 1;
  exitCalibrationMode();
  const next = currentViewScanRecommendation();
  if (next) seekToFrame(next.frame);
  updateViewScanWorkflow();
}

function atlasCoverageSnapshot(atlas = currentCameraAtlas()) {
  const coverage = atlas?.coverage_summary || {};
  return {
    strongAnchors: Number(atlas?.summary?.strong_anchor_count || 0),
    median: coverage.coverage_median_px ?? null,
    p90: coverage.coverage_p90_px ?? null,
    max: coverage.coverage_max_px ?? null,
    thresholds: coverage.threshold_coverage || {},
    recommendations: (anchorSelectionResult?.video === selectedVideo?.name
      ? anchorSelectionResult.result?.selection?.recommendations
      : []) || [],
  };
}

function activeGuidedRecommendation() {
  return guidedCalibration?.recommendation || null;
}

function nextGuidedRecommendation(selection) {
  const skipped = guidedCalibration?.skippedFrames || new Set();
  return (selection?.recommendations || []).find((item) => !skipped.has(item.frame)) || null;
}

function formatGuidedNumber(value, suffix = "px") {
  return Number.isFinite(Number(value)) ? `${Number(value).toFixed(1)} ${suffix}` : "—";
}

function updateGuidedCalibrationPanel() {
  const active = Boolean(guidedCalibration?.active && activeMode === "field");
  guidedCalibrationPanel.hidden = !active;
  if (!active) return;

  const fragment = document.createDocumentFragment();
  const add = (text, emphasis = false, className = "") => {
    const element = document.createElement(emphasis ? "strong" : "span");
    if (className) element.className = className;
    element.textContent = text;
    fragment.append(element);
  };
  const recommendation = activeGuidedRecommendation();
  if (guidedCalibration.running) {
    add("Analizando vistas…", true);
  } else if (recommendation) {
    const view = recommendation.view || {};
    const nearest = recommendation.nearest_anchor || {};
    add(`Frame recomendado: ${recommendation.frame}`, true);
    add(`Cobertura: ${formatGuidedNumber(recommendation.coverage_distance_px)} · Calidad visual: ${(Number(recommendation.quality_score || 0) * 100).toFixed(0)} %`);
    add(`Vista: ${view.side || "—"} · Zoom: ${view.zoom || "—"} · Campo visible: ${(Number(view.visible_fraction || 0) * 100).toFixed(0)} %`);
    add(`Líneas visibles: ${view.visible_lines ?? 0}${nearest.frame !== undefined ? ` · HUMAN más cercano: ${nearest.frame} (${formatGuidedNumber(nearest.median_px)})` : ""}`);
  } else if (guidedCalibration.selectorError) {
    add(`Selector no disponible: ${guidedCalibration.selectorError}`, true, "guided-warning");
  } else if (guidedCalibration.allRecommendationsSkipped) {
    add("Se han omitido todos los candidatos disponibles en esta sesión.", true);
  } else if (guidedCalibration.selectorSucceeded) {
    add("No quedan candidatos prioritarios detectados.", true);
  } else {
    add("Sin recomendación disponible.", true);
  }

  const impact = guidedCalibration.impact;
  if (impact) {
    add("IMPACTO DEL ANCHOR", true, "guided-heading");
    add(`HUMAN anchors: ${impact.before.strongAnchors} → ${impact.after.strongAnchors}`);
    add(`Cobertura mediana: ${formatGuidedNumber(impact.before.median)} → ${formatGuidedNumber(impact.after.median)}`);
    add(`P90: ${formatGuidedNumber(impact.before.p90)} → ${formatGuidedNumber(impact.after.p90)} · Máximo: ${formatGuidedNumber(impact.before.max)} → ${formatGuidedNumber(impact.after.max)}`);
    if (impact.disappeared.length) add(`Desaparecen: ${impact.disappeared.join(", ")}`);
    if (impact.lowered.length) add(`Bajan de prioridad: ${impact.lowered.join(", ")}`);
    if (impact.emerged.length) add(`Nuevas: ${impact.emerged.join(", ")}`);
  }

  const coverage = currentCameraAtlas()?.coverage_summary;
  if (guidedCalibration.checked && coverage) {
    add("COBERTURA FIELD", true, "guided-heading");
    add(`HUMAN: ${coverage.anchor_count ?? 0} · Mediana: ${formatGuidedNumber(coverage.coverage_median_px)} · P90: ${formatGuidedNumber(coverage.coverage_p90_px)} · Máximo: ${formatGuidedNumber(coverage.coverage_max_px)}`);
    const thresholds = coverage.threshold_coverage || {};
    add(`≤50: ${(Number(thresholds["50"] || 0) * 100).toFixed(0)} % · ≤100: ${(Number(thresholds["100"] || 0) * 100).toFixed(0)} % · ≤200: ${(Number(thresholds["200"] || 0) * 100).toFixed(0)} % · ≤400: ${(Number(thresholds["400"] || 0) * 100).toFixed(0)} %`);
    add(`Candidatos prioritarios: ${(anchorSelectionResult?.result?.selection?.recommendations || []).length}`);
  }

  const reviews = anchorSelectionResult?.result?.selection?.isolated_anchor_reviews || [];
  if (reviews.length) {
    add("ANCHORS HUMAN A REVISAR", true, "guided-heading");
    reviews.forEach((review) => {
      const button = document.createElement("button");
      button.type = "button";
      button.dataset.guidedReviewFrame = review.frame;
      button.textContent = `Anchor HUMAN aislado: frame ${review.frame} · cercano ${review.nearest_anchor_frame} (${formatGuidedNumber(review.nearest_anchor_distance_px)})`;
      fragment.append(button);
    });
  }
  guidedCalibrationContent.replaceChildren(fragment);
  guidedContextButton.disabled = !recommendation || guidedCalibration.running;
  guidedReturnButton.disabled = !recommendation || guidedCalibration.running;
  guidedSkipButton.disabled = !recommendation || guidedCalibration.running;
  const currentIsSolvedRecommendation = recommendation
    && currentFrameNumber() === recommendation.frame
    && hasSolvedManualCalibration(calibrationForCurrentFrame())
    && !calibrationSolvePending;
  guidedConfirmButton.disabled = !currentIsSolvedRecommendation || guidedCalibration.running;
  guidedContinueButton.hidden = !impact;
  guidedContinueButton.disabled = !recommendation || guidedCalibration.running;
}

function comparisonImpact(before, after) {
  const beforeRanks = new Map(before.recommendations.map((item) => [item.frame, item.rank]));
  const afterRanks = new Map(after.recommendations.map((item) => [item.frame, item.rank]));
  return {
    before,
    after,
    disappeared: [...beforeRanks.keys()].filter((frame) => !afterRanks.has(frame)),
    lowered: [...beforeRanks].filter(([frame, rank]) => afterRanks.has(frame) && afterRanks.get(frame) >= rank + 2).map(([frame]) => frame),
    emerged: [...afterRanks.keys()].filter((frame) => !beforeRanks.has(frame)),
  };
}

async function refreshGuidedRecommendations({ seek = false, impactBefore = null, checked = false } = {}) {
  if (!guidedCalibration?.active) return;
  // Invalidate both the in-memory result and its guided interpretation before
  // rebuilding Atlas. This prevents stale rankings from a previous review
  // state from becoming a recommendation after a recalculation.
  anchorSelectionResult = null;
  anchorSelectionError = null;
  guidedCalibration.recommendation = null;
  guidedCalibration.selectorError = null;
  guidedCalibration.selectorSucceeded = false;
  guidedCalibration.allRecommendationsSkipped = false;
  guidedCalibration.running = true;
  updateGuidedCalibrationPanel();
  const atlas = await analyzeCameraAtlas();
  if (!guidedCalibration?.active) return;
  const selectorResponse = await requestAnchorSelection();
  if (!guidedCalibration?.active) return;
  guidedCalibration.checked = checked || guidedCalibration.checked;
  if (!selectorResponse?.ok) {
    guidedCalibration.running = false;
    guidedCalibration.selectorError = selectorResponse?.error || anchorSelectionError || "No se pudo ejecutar el selector de anchors.";
    updateGuidedCalibrationPanel();
    return;
  }
  const selection = selectorResponse.result.selection;
  const after = atlasCoverageSnapshot(atlas);
  guidedCalibration.recommendation = nextGuidedRecommendation(selection);
  guidedCalibration.selectorSucceeded = true;
  guidedCalibration.allRecommendationsSkipped = !guidedCalibration.recommendation && selection.recommendations.length > 0;
  guidedCalibration.running = false;
  if (impactBefore) guidedCalibration.impact = comparisonImpact(impactBefore, after);
  else if (guidedCalibration.recommendation) guidedCalibration.preAnchorSnapshot = after;
  updateGuidedCalibrationPanel();
  if (seek && guidedCalibration.recommendation) seekToFrame(guidedCalibration.recommendation.frame);
}

async function startGuidedCalibration() {
  if (!selectedVideo) return;
  video.pause();
  guidedCalibration = { active: true, running: false, skippedFrames: new Set(), recommendation: null, impact: null, checked: false };
  await refreshGuidedRecommendations({ seek: true });
}

function exitGuidedCalibration() {
  if (!guidedCalibration) return;
  guidedCalibration = null;
  updateGuidedCalibrationPanel();
}

function playGuidedContext() {
  const recommendation = activeGuidedRecommendation();
  if (!recommendation || !selectedVideo?.fps) return;
  const contextSeconds = 3;
  guidedCalibration.contextEndTime = Math.min((recommendation.frame / selectedVideo.fps) + contextSeconds, video.duration || Infinity);
  video.currentTime = Math.max(0, (recommendation.frame / selectedVideo.fps) - contextSeconds);
  video.play();
}

function returnToGuidedRecommendation() {
  const recommendation = activeGuidedRecommendation();
  if (recommendation) seekToFrame(recommendation.frame);
}

async function skipGuidedRecommendation() {
  const recommendation = activeGuidedRecommendation();
  if (!recommendation || !guidedCalibration) return;
  guidedCalibration.skippedFrames.add(recommendation.frame);
  guidedCalibration.impact = null;
  await refreshGuidedRecommendations({ seek: true });
}

async function confirmGuidedAnchor() {
  const recommendation = activeGuidedRecommendation();
  if (!recommendation || currentFrameNumber() !== recommendation.frame || !hasSolvedManualCalibration(calibrationForCurrentFrame())) return;
  const before = guidedCalibration.preAnchorSnapshot || atlasCoverageSnapshot();
  guidedCalibration.preAnchorSnapshot = null;
  await refreshGuidedRecommendations({ impactBefore: before });
}

function cloneAnnotations() {
  return JSON.parse(JSON.stringify(fieldAnnotations));
}

function cloneFieldState() {
  return JSON.parse(JSON.stringify({
    annotations: fieldAnnotations,
    calibrations: fieldCalibrations,
    propagations: fieldPropagations,
    refinedSegments: fieldRefinedSegments,
  }));
}

function rememberFieldEdit() {
  fieldUndoStack.push(cloneFieldState());
  setReviewDirty(true);
}

function invalidatePropagationsFromAnchor(calibration) {
  if (!calibration) return;
  fieldPropagations = fieldPropagations.filter((propagation) => !(
    propagation.video === calibration.video && propagation.anchorFrame === calibration.frame
  ));
}

function invalidateRefinedSegmentsForCalibration(calibration) {
  if (!calibration) return;
  fieldRefinedSegments = fieldRefinedSegments.filter((segment) => !(
    segment.video === calibration.video
    && calibration.frame >= segment.leftAnchorFrame
    && calibration.frame <= segment.rightAnchorFrame
  ));
}

function clampToVideo({ x, y }) {
  return {
    x: Math.min(Math.max(x, 0), video.videoWidth),
    y: Math.min(Math.max(y, 0), video.videoHeight),
  };
}

function createFieldAnnotation(type, coordinates, fit = null) {
  annotationSequence += 1;
  const annotation = {
    id: `field-${Date.now()}-${annotationSequence}`,
    video: selectedVideo.name,
    frame: currentFrameNumber(),
    type,
    coordinates,
    source: "manual",
  };
  if (fit) annotation.fit = fit;
  return annotation;
}

function findFieldAnnotation(id) {
  return fieldAnnotations.find((annotation) => annotation.id === id);
}

function distanceBetween(first, second) {
  return Math.hypot(first.x - second.x, first.y - second.y);
}

function distanceToLineSegment(point, start, end) {
  const deltaX = end.x - start.x;
  const deltaY = end.y - start.y;
  const lengthSquared = deltaX * deltaX + deltaY * deltaY;
  if (!lengthSquared) return distanceBetween(point, start);
  const ratio = Math.max(0, Math.min(1, ((point.x - start.x) * deltaX + (point.y - start.y) * deltaY) / lengthSquared));
  return distanceBetween(point, { x: start.x + ratio * deltaX, y: start.y + ratio * deltaY });
}

function hitTestFieldAnnotation(coordinates) {
  if (!displayedVideoRect) return null;
  const hitRadius = 12 / (displayedVideoRect.width / video.videoWidth);
  const annotations = annotationsForCurrentFrame();
  for (let index = annotations.length - 1; index >= 0; index -= 1) {
    const annotation = annotations[index];
    if (annotation.type === "point" && distanceBetween(coordinates, annotation.coordinates) <= hitRadius) {
      return { id: annotation.id, handle: "point" };
    }
    if (annotation.type === "line") {
      if (distanceBetween(coordinates, annotation.coordinates.start) <= hitRadius) return { id: annotation.id, handle: "start" };
      if (distanceBetween(coordinates, annotation.coordinates.end) <= hitRadius) return { id: annotation.id, handle: "end" };
      if (distanceToLineSegment(coordinates, annotation.coordinates.start, annotation.coordinates.end) <= hitRadius) {
        return { id: annotation.id, handle: "line" };
      }
    }
    if (annotation.type === "ellipse") {
      const { center, radiusX, radiusY, rotation } = annotation.coordinates;
      const cos = Math.cos(rotation);
      const sin = Math.sin(rotation);
      const xHandle = { x: center.x + radiusX * cos, y: center.y + radiusX * sin };
      const yHandle = { x: center.x - radiusY * sin, y: center.y + radiusY * cos };
      if (distanceBetween(coordinates, center) <= hitRadius) return { id: annotation.id, handle: "ellipse-center" };
      if (distanceBetween(coordinates, xHandle) <= hitRadius) return { id: annotation.id, handle: "ellipse-radius-x" };
      if (distanceBetween(coordinates, yHandle) <= hitRadius) return { id: annotation.id, handle: "ellipse-radius-y" };

      const relativeX = coordinates.x - center.x;
      const relativeY = coordinates.y - center.y;
      const localX = relativeX * cos + relativeY * sin;
      const localY = -relativeX * sin + relativeY * cos;
      const normalizedRadius = Math.hypot(localX / radiusX, localY / radiusY);
      if (Math.abs(normalizedRadius - 1) * Math.min(radiusX, radiusY) <= hitRadius) {
        return { id: annotation.id, handle: "ellipse" };
      }
    }
  }
  return null;
}

function fitSmartLine(samples) {
  const center = samples.reduce((total, point) => ({ x: total.x + point.x, y: total.y + point.y }), { x: 0, y: 0 });
  center.x /= samples.length;
  center.y /= samples.length;
  let covarianceXX = 0;
  let covarianceXY = 0;
  let covarianceYY = 0;
  samples.forEach((point) => {
    const x = point.x - center.x;
    const y = point.y - center.y;
    covarianceXX += x * x;
    covarianceXY += x * y;
    covarianceYY += y * y;
  });
  const rotation = 0.5 * Math.atan2(2 * covarianceXY, covarianceXX - covarianceYY);
  const direction = { x: Math.cos(rotation), y: Math.sin(rotation) };
  const projections = samples.map((point) => (
    (point.x - center.x) * direction.x + (point.y - center.y) * direction.y
  ));
  return {
    start: clampToVideo({ x: center.x + Math.min(...projections) * direction.x, y: center.y + Math.min(...projections) * direction.y }),
    end: clampToVideo({ x: center.x + Math.max(...projections) * direction.x, y: center.y + Math.max(...projections) * direction.y }),
  };
}

function ellipseResidual(parameters, samples) {
  const [centerX, centerY, radiusX, radiusY, rotation] = parameters;
  const cos = Math.cos(rotation);
  const sin = Math.sin(rotation);
  return samples.reduce((error, point) => {
    const relativeX = point.x - centerX;
    const relativeY = point.y - centerY;
    const localX = relativeX * cos + relativeY * sin;
    const localY = -relativeX * sin + relativeY * cos;
    const radialError = Math.hypot(localX / radiusX, localY / radiusY) - 1;
    return error + radialError * radialError;
  }, 0) / samples.length;
}

function fitEllipse(samples) {
  const center = samples.reduce((total, point) => ({ x: total.x + point.x, y: total.y + point.y }), { x: 0, y: 0 });
  center.x /= samples.length;
  center.y /= samples.length;
  let covarianceXX = 0;
  let covarianceXY = 0;
  let covarianceYY = 0;
  samples.forEach((point) => {
    const x = point.x - center.x;
    const y = point.y - center.y;
    covarianceXX += x * x;
    covarianceXY += x * y;
    covarianceYY += y * y;
  });
  const rotation = 0.5 * Math.atan2(2 * covarianceXY, covarianceXX - covarianceYY);
  const cos = Math.cos(rotation);
  const sin = Math.sin(rotation);
  const localPoints = samples.map((point) => ({
    x: (point.x - center.x) * cos + (point.y - center.y) * sin,
    y: -(point.x - center.x) * sin + (point.y - center.y) * cos,
  }));
  const xValues = localPoints.map((point) => point.x);
  const yValues = localPoints.map((point) => point.y);
  const localCenter = { x: (Math.min(...xValues) + Math.max(...xValues)) / 2, y: (Math.min(...yValues) + Math.max(...yValues)) / 2 };
  let parameters = [
    center.x + localCenter.x * cos - localCenter.y * sin,
    center.y + localCenter.x * sin + localCenter.y * cos,
    Math.max((Math.max(...xValues) - Math.min(...xValues)) / 2, 8),
    Math.max((Math.max(...yValues) - Math.min(...yValues)) / 2, 8),
    rotation,
  ];
  let steps = [Math.max(parameters[2] * 0.08, 4), Math.max(parameters[3] * 0.08, 4), Math.max(parameters[2] * 0.08, 3), Math.max(parameters[3] * 0.08, 3), 0.04];
  let bestError = ellipseResidual(parameters, samples);

  for (let iteration = 0; iteration < 55; iteration += 1) {
    let improved = false;
    for (let index = 0; index < parameters.length; index += 1) {
      let bestParameters = parameters;
      for (const direction of [-1, 1]) {
        const candidate = [...parameters];
        candidate[index] += direction * steps[index];
        candidate[0] = Math.min(Math.max(candidate[0], -video.videoWidth), video.videoWidth * 2);
        candidate[1] = Math.min(Math.max(candidate[1], -video.videoHeight), video.videoHeight * 2);
        candidate[2] = Math.min(Math.max(candidate[2], 6), video.videoWidth * 2);
        candidate[3] = Math.min(Math.max(candidate[3], 6), video.videoHeight * 2);
        const error = ellipseResidual(candidate, samples);
        if (error < bestError) {
          bestError = error;
          bestParameters = candidate;
          improved = true;
        }
      }
      parameters = bestParameters;
    }
    if (!improved) steps = steps.map((step) => step * 0.82);
  }

  const [centerX, centerY, radiusX, radiusY, fittedRotation] = parameters;
  let previousAngle = null;
  const unwrappedAngles = samples.map((point) => {
    const relativeX = point.x - centerX;
    const relativeY = point.y - centerY;
    let angle = Math.atan2(
      (-relativeX * Math.sin(fittedRotation) + relativeY * Math.cos(fittedRotation)) / radiusY,
      (relativeX * Math.cos(fittedRotation) + relativeY * Math.sin(fittedRotation)) / radiusX,
    );
    if (previousAngle !== null) angle += Math.round((previousAngle - angle) / (Math.PI * 2)) * Math.PI * 2;
    previousAngle = angle;
    return angle;
  });
  const arc = {
    startAngle: unwrappedAngles[0],
    endAngle: unwrappedAngles[unwrappedAngles.length - 1],
    counterclockwise: unwrappedAngles[unwrappedAngles.length - 1] < unwrappedAngles[0],
  };
  return { center: { x: centerX, y: centerY }, radiusX, radiusY, rotation: fittedRotation, arc };
}

function setActiveFieldTool(tool) {
  activeFieldTool = tool;
  setCalibrationMode(false);
  calibrationPanel.hidden = true;
  fieldToolButtons.forEach((button) => button.classList.toggle("is-active", button.dataset.fieldTool === tool));
  lineDraft = null;
  assistedDraft = null;
  calibrationLineDraft = null;
  calibrationArcDraft = null;
  if (video.readyState >= HTMLMediaElement.HAVE_METADATA) video.pause();
  drawOverlay();
}

function undoFieldEdit() {
  const previousState = fieldUndoStack.pop();
  if (!previousState) return;
  fieldAnnotations = previousState.annotations;
  fieldCalibrations = previousState.calibrations;
  fieldPropagations = previousState.propagations || [];
  fieldRefinedSegments = previousState.refinedSegments || [];
  selectedFieldItem = null;
  lineDraft = null;
  assistedDraft = null;
  calibrationDragState = null;
  calibrationLineDraft = null;
  calibrationArcDraft = null;
  setReviewDirty(true);
  updateCalibrationStatus();
  drawOverlay();
}

function deleteSelectedFieldItem() {
  if (activeMode !== "guided" || !selectedFieldItem) return;
  const annotation = findFieldAnnotation(selectedFieldItem.id);
  if (!annotation) return;
  rememberFieldEdit();
  fieldAnnotations = fieldAnnotations.filter((item) => item.id !== annotation.id);
  selectedFieldItem = null;
  drawOverlay();
}

function calibrationConstraints(calibration) {
  if (!calibration) return [];
  if (!Array.isArray(calibration.constraints)) {
    calibration.constraints = (calibration.correspondences || []).map((item) => ({
      type: "point", semantic: item.landmark, world: item.world, image: item.image, source: "manual",
    }));
  }
  return calibration.constraints;
}

function syncLegacyPointCorrespondences(calibration) {
  calibration.correspondences = calibrationConstraints(calibration)
    .filter((constraint) => constraint.type === "point")
    .map((constraint) => ({ landmark: constraint.semantic, world: constraint.world, image: constraint.image }));
}

function semanticDefinitionsForType(type = calibrationType) {
  if (type === "point") return PITCH_MODEL.landmarks;
  if (type === "line") return PITCH_MODEL.calibrationLines;
  if (type === "arc") return PITCH_MODEL.calibrationArcs;
  return [];
}

function syncSemanticSelector() {
  const definitions = semanticDefinitionsForType();
  if (!definitions.some((item) => item.id === calibrationSemantic)) {
    calibrationSemantic = selectedSemanticByType[calibrationType];
  }
  if (!definitions.some((item) => item.id === calibrationSemantic)) {
    calibrationSemantic = definitions[0]?.id || null;
  }
  selectedSemanticByType[calibrationType] = calibrationSemantic;
  calibrationLandmark.replaceChildren(...definitions.map((item) => new Option(item.label, item.id)));
  calibrationLandmark.value = calibrationSemantic || "";
}

function setCalibrationSelection(type, semantic = selectedSemanticByType[type]) {
  calibrationType = type;
  calibrationConstraintType.value = type;
  calibrationSemantic = semantic;
  syncSemanticSelector();
}

function selectedSemanticDefinition() {
  if (calibrationType === "point") return PITCH_MODEL.landmarksById[calibrationSemantic];
  if (calibrationType === "line") return PITCH_MODEL.calibrationLinesById[calibrationSemantic];
  if (calibrationType === "arc") return PITCH_MODEL.calibrationArcsById[calibrationSemantic];
  return null;
}

const LINE_WARNING_ERROR_PX = 12;

function lineConstraintError(constraint, homography, samples = 12) {
  const { start: worldStart, end: worldEnd } = constraint.world;
  const { start: imageStart, end: imageEnd } = constraint.image;
  const imageLength = Math.hypot(imageEnd.x - imageStart.x, imageEnd.y - imageStart.y);
  if (!Array.isArray(homography) || imageLength < 1e-6) return null;

  const errors = [];
  for (let index = 0; index <= samples; index += 1) {
    const ratio = index / samples;
    const projected = projectWorldPoint({
      x: worldStart.x + (worldEnd.x - worldStart.x) * ratio,
      y: worldStart.y + (worldEnd.y - worldStart.y) * ratio,
    }, homography);
    if (!projected) return null;
    const signedDistance = (
      (imageEnd.x - imageStart.x) * (imageStart.y - projected.y)
      - (imageStart.x - projected.x) * (imageEnd.y - imageStart.y)
    ) / imageLength;
    errors.push(Math.abs(signedDistance));
  }
  return errors.length ? errors.reduce((sum, value) => sum + value, 0) / errors.length : null;
}

function calibrationLineDiagnostics(calibration) {
  if (!hasSolvedManualCalibration(calibration)) return [];
  const solverDiagnostics = calibration.diagnostics?.line_errors;
  if (Array.isArray(solverDiagnostics)) {
    return solverDiagnostics
      .filter((diagnostic) => Number.isFinite(diagnostic.error_px))
      .map((diagnostic) => ({
        semantic: diagnostic.semantic,
        label: PITCH_MODEL.calibrationLinesById[diagnostic.semantic]?.label || diagnostic.semantic,
        errorPx: diagnostic.error_px,
        quality: diagnostic.quality,
      }));
  }
  return calibrationConstraints(calibration)
    .filter((constraint) => constraint.type === "line")
    .map((constraint) => ({
      semantic: constraint.semantic,
      label: PITCH_MODEL.calibrationLinesById[constraint.semantic]?.label || constraint.semantic,
      errorPx: lineConstraintError(constraint, calibration.homography),
      quality: null,
    }))
    .filter((diagnostic) => Number.isFinite(diagnostic.errorPx));
}

function calibrationStrategyLabel(calibration) {
  const strategy = calibration?.diagnostics?.calibration_strategy;
  if (!strategy?.selected_mode) return "";
  const selected = strategy.selected_mode;
  if (!debugOverlayEnabled) return selected === "full" ? "Calibración completa" : "Corrección parcial";

  const evidence = strategy.evidence || {};
  if (selected === "full") {
    const score = Number(evidence.evidence_score || 0).toFixed(2);
    return `FULL · ${evidence.points || 0} puntos · ${evidence.lines || 0} líneas · evidencia ${score}`;
  }
  return "REFINE · evidencia parcial";
}

function constraintCountsLabel(constraints) {
  const counts = { point: 0, line: 0, arc: 0 };
  constraints.forEach((constraint) => { if (counts[constraint.type] !== undefined) counts[constraint.type] += 1; });
  const labels = [
    counts.point && `${counts.point} ${counts.point === 1 ? "punto" : "puntos"}`,
    counts.line && `${counts.line} ${counts.line === 1 ? "línea" : "líneas"}`,
    counts.arc && `${counts.arc} ${counts.arc === 1 ? "arco" : "arcos"}`,
  ].filter(Boolean);
  return labels.join(" · ") || "0 referencias";
}

function updateCurrentReferenceLabel() {
  if (!currentReferenceLabel) return;
  const semantic = selectedSemanticDefinition();
  const calibration = calibrationForCurrentFrame();
  const selectedOnCurrentFrame = selectedCalibrationConstraint?.calibrationId === calibration?.id;
  const shouldShow = Boolean(semantic && (calibrationMode || selectedOnCurrentFrame || calibrationLineDraft || calibrationArcDraft));
  const warning = calibrationLineDiagnostics(calibration).find((diagnostic) => (
    diagnostic.quality === "bad" || (!diagnostic.quality && diagnostic.errorPx > LINE_WARNING_ERROR_PX)
  ));
  currentReferenceLabel.hidden = !shouldShow && !warning;
  currentReferenceLabel.textContent = shouldShow
    ? `Referencia actual: ${semantic.label}`
    : warning
      ? `${warning.label}: error ${Math.round(warning.errorPx)} px`
      : "";
}

function renderCalibrationPitchGeometry() {
  const svgNamespace = "http://www.w3.org/2000/svg";
  const horizontalMargin = Math.max(5, PITCH_MODEL.lengthM * 0.05);
  const verticalMargin = Math.max(9, PITCH_MODEL.widthM * 0.13);
  calibrationPitch.setAttribute(
    "viewBox",
    `${-PITCH_MODEL.halfLength - horizontalMargin} ${-PITCH_MODEL.halfWidth - verticalMargin} ${PITCH_MODEL.lengthM + horizontalMargin * 2} ${PITCH_MODEL.widthM + verticalMargin * 2}`,
  );
  calibrationPitchGeometry.replaceChildren();
  PITCH_MODEL.segments.forEach((segment) => {
    const path = document.createElementNS(svgNamespace, "polyline");
    path.setAttribute("points", segment.map((point) => `${point.x},${point.y}`).join(" "));
    path.classList.add("pitch-diagram-line");
    calibrationPitchGeometry.append(path);
  });
  PITCH_MODEL.calibrationArcs.forEach((arc) => {
    const { center, radius, startAngle, endAngle } = arc.world;
    const element = document.createElementNS(svgNamespace, Number.isFinite(startAngle) ? "path" : "circle");
    element.classList.add("pitch-diagram-line");
    if (Number.isFinite(startAngle) && Number.isFinite(endAngle)) {
      const start = { x: center.x + radius * Math.cos(startAngle), y: center.y + radius * Math.sin(startAngle) };
      const end = { x: center.x + radius * Math.cos(endAngle), y: center.y + radius * Math.sin(endAngle) };
      const largeArc = Math.abs(endAngle - startAngle) > Math.PI ? 1 : 0;
      const sweep = endAngle >= startAngle ? 1 : 0;
      element.setAttribute("d", `M ${start.x} ${start.y} A ${radius} ${radius} 0 ${largeArc} ${sweep} ${end.x} ${end.y}`);
    } else {
      element.setAttribute("cx", center.x);
      element.setAttribute("cy", center.y);
      element.setAttribute("r", radius);
    }
    calibrationPitchGeometry.append(element);
  });
  pitchDiagramTopLabel.setAttribute("x", "0");
  pitchDiagramTopLabel.setAttribute("y", String(-PITCH_MODEL.halfWidth - verticalMargin * 0.45));
  pitchDiagramLeftLabel.setAttribute("x", String(-PITCH_MODEL.halfLength - horizontalMargin * 0.9));
  pitchDiagramLeftLabel.setAttribute("y", String(PITCH_MODEL.halfWidth + verticalMargin * 0.5));
  pitchDiagramRightLabel.setAttribute("x", String(PITCH_MODEL.halfLength + horizontalMargin * 0.9));
  pitchDiagramRightLabel.setAttribute("y", String(PITCH_MODEL.halfWidth + verticalMargin * 0.5));
}

function renderCalibrationPitch() {
  const svgNamespace = "http://www.w3.org/2000/svg";
  const calibration = calibrationForCurrentFrame();
  const constraints = calibrationConstraints(calibration);
  renderCalibrationPitchGeometry();
  calibrationPitchMarkers.replaceChildren();
  calibrationPitchLines.replaceChildren();
  calibrationPitchArcs.replaceChildren();

  PITCH_MODEL.landmarks.forEach((landmark) => {
    const assignedIndex = constraints.findIndex((item) => item.type === "point" && item.semantic === landmark.id);
    const marker = document.createElementNS(svgNamespace, "g");
    marker.classList.add("pitch-landmark");
    if (calibrationType !== "point") marker.style.display = "none";
    if (landmark.id === calibrationSemantic) marker.classList.add("is-selected");
    if (assignedIndex >= 0) marker.classList.add("is-assigned");
    marker.setAttribute("transform", `translate(${landmark.world.x} ${landmark.world.y})`);
    marker.setAttribute("data-landmark", landmark.id);
    marker.setAttribute("aria-label", landmark.label);
    marker.setAttribute("role", "button");

    const circle = document.createElementNS(svgNamespace, "circle");
    circle.setAttribute("r", "2.2");
    const title = document.createElementNS(svgNamespace, "title");
    title.textContent = landmark.label;
    marker.append(circle, title);
    if (assignedIndex >= 0) {
      const number = document.createElementNS(svgNamespace, "text");
      number.textContent = String(assignedIndex + 1);
      marker.append(number);
    }
    marker.addEventListener("click", () => {
      setCalibrationSelection("point", landmark.id);
      updateCalibrationStatus();
      drawOverlay();
    });
    calibrationPitchMarkers.append(marker);
  });

  PITCH_MODEL.calibrationLines.forEach((line) => {
    const assignedIndex = constraints.findIndex((item) => item.type === "line" && item.semantic === line.id);
    const selector = document.createElementNS(svgNamespace, "line");
    selector.setAttribute("x1", line.world.start.x);
    selector.setAttribute("y1", line.world.start.y);
    selector.setAttribute("x2", line.world.end.x);
    selector.setAttribute("y2", line.world.end.y);
    selector.classList.add("pitch-line-selector");
    selector.style.pointerEvents = calibrationType === "line" ? "stroke" : "none";
    if (calibrationType === "line") selector.classList.add("is-visible");
    if (line.id === calibrationSemantic) selector.classList.add("is-selected");
    if (assignedIndex >= 0) selector.classList.add("is-assigned");
    selector.addEventListener("click", () => {
      setCalibrationSelection("line", line.id);
      updateCalibrationStatus();
      drawOverlay();
    });
    calibrationPitchLines.append(selector);
    if (assignedIndex >= 0) {
      const number = document.createElementNS(svgNamespace, "text");
      number.classList.add("pitch-constraint-number");
      number.setAttribute("x", (line.world.start.x + line.world.end.x) / 2);
      number.setAttribute("y", (line.world.start.y + line.world.end.y) / 2);
      number.textContent = String(assignedIndex + 1);
      calibrationPitchLines.append(number);
    }
  });

  PITCH_MODEL.calibrationArcs.forEach((arc) => {
    const assignedIndex = constraints.findIndex((item) => item.type === "arc" && item.semantic === arc.id);
    const hasArcSpan = Number.isFinite(arc.world.startAngle) && Number.isFinite(arc.world.endAngle);
    const selector = document.createElementNS(svgNamespace, hasArcSpan ? "path" : "circle");
    if (hasArcSpan) {
      const { center, radius, startAngle, endAngle } = arc.world;
      const start = {
        x: center.x + radius * Math.cos(startAngle),
        y: center.y + radius * Math.sin(startAngle),
      };
      const end = {
        x: center.x + radius * Math.cos(endAngle),
        y: center.y + radius * Math.sin(endAngle),
      };
      const largeArc = Math.abs(endAngle - startAngle) > Math.PI ? 1 : 0;
      const sweep = endAngle >= startAngle ? 1 : 0;
      selector.setAttribute("d", `M ${start.x} ${start.y} A ${radius} ${radius} 0 ${largeArc} ${sweep} ${end.x} ${end.y}`);
    } else {
      selector.setAttribute("cx", arc.world.center.x);
      selector.setAttribute("cy", arc.world.center.y);
      selector.setAttribute("r", arc.world.radius);
    }
    selector.classList.add("pitch-arc-selector");
    selector.style.pointerEvents = calibrationType === "arc" ? "stroke" : "none";
    if (calibrationType === "arc") selector.classList.add("is-visible");
    if (arc.id === calibrationSemantic) selector.classList.add("is-selected");
    if (assignedIndex >= 0) selector.classList.add("is-assigned");
    selector.addEventListener("click", () => {
      setCalibrationSelection("arc", arc.id);
      updateCalibrationStatus();
      drawOverlay();
    });
    const title = document.createElementNS(svgNamespace, "title");
    title.textContent = arc.label;
    selector.append(title);
    calibrationPitchArcs.append(selector);
    if (assignedIndex >= 0) {
      const number = document.createElementNS(svgNamespace, "text");
      number.classList.add("pitch-constraint-number");
      number.setAttribute("x", arc.world.center.x);
      number.setAttribute("y", arc.world.center.y - arc.world.radius - 2);
      number.textContent = String(assignedIndex + 1);
      calibrationPitchArcs.append(number);
    }
  });
}

function calibrationSolveIsPending(calibration = calibrationForCurrentFrame()) {
  return Boolean(calibration && calibrationSolvePending?.calibrationId === calibration.id);
}

function calibrationSolveErrorFor(calibration = calibrationForCurrentFrame()) {
  return calibration && calibrationSolveError?.calibrationId === calibration.id
    ? calibrationSolveError.message
    : null;
}

function updateCalibrationStatus(message = null) {
  syncSemanticSelector();
  const calibration = calibrationForCurrentFrame();
  const pending = calibrationSolveIsPending(calibration);
  const solveError = calibrationSolveErrorFor(calibration);
  calibrationSolverLoading.hidden = !pending;
  exitCalibrationButton.disabled = Boolean(calibrationSolvePending);
  const refiningExistingProjection = hasInitialHomography(calibration) || (!calibration && Boolean(existingProjectionForCurrentFrame()));
  renderCalibrationPitch();
  updateCurrentReferenceLabel();
  const drawingHint = calibrationType === "point"
    ? "Elige un punto en este diagrama y después haz clic en el mismo punto del vídeo."
    : calibrationType === "line"
      ? "Elige una línea del diagrama y arrastra un tramo visible de esa línea en el vídeo."
      : "Elige el arco del diagrama y traza su parte visible en el vídeo.";
  calibrationHint.textContent = refiningExistingProjection
    ? `${drawingHint} Corrigiendo la proyección existente de este fotograma.`
    : drawingHint;
  if (pending) {
    calibrationStatus.textContent = constraintCountsLabel(calibrationConstraints(calibration));
    return;
  }
  if (message) {
    calibrationStatus.textContent = `${constraintCountsLabel(calibrationConstraints(calibration))}${refiningExistingProjection ? " · corrigiendo proyección existente" : ""} · ${message}`;
    return;
  }
  if (solveError) {
    calibrationStatus.textContent = `${constraintCountsLabel(calibrationConstraints(calibration))} · Error: ${solveError}`;
    return;
  }
  const constraints = calibrationConstraints(calibration);
  const lineWarning = calibrationLineDiagnostics(calibration).find((diagnostic) => (
    diagnostic.quality === "bad" || (!diagnostic.quality && diagnostic.errorPx > LINE_WARNING_ERROR_PX)
  ));
  const strategyLabel = calibrationStrategyLabel(calibration);
  const hasMixedConstraints = constraints.some((item) => item.type !== "point");
  calibrationStatus.textContent = calibration?.homography
    ? `${constraintCountsLabel(constraints)} · campo proyectado${strategyLabel ? ` · ${strategyLabel}` : ""}${lineWarning ? ` · ${lineWarning.label} no está siendo respetada (error: ${Math.round(lineWarning.errorPx)} px)` : ""}`
    : refiningExistingProjection
      ? `${constraintCountsLabel(constraints)} · corrigiendo proyección existente`
      : hasMixedConstraints
      ? `${constraintCountsLabel(constraints)} · solver mixto pendiente`
      : `${constraintCountsLabel(constraints)} / mínimo 4`;
  updatePitchSetupWorkflow();
}

function enterCalibrationMode() {
  if (!selectedVideo) return;
  video.pause();
  setCalibrationMode(true);
  activeFieldTool = "select";
  fieldToolButtons.forEach((button) => button.classList.remove("is-active"));
  calibrationPanel.hidden = false;
  positionCalibrationPanel();
  updateCalibrationStatus();
  updatePitchSetupWorkflow();
  drawOverlay();
}

function exitCalibrationMode() {
  setCalibrationMode(false);
  calibrationDragState = null;
  calibrationLineDraft = null;
  calibrationArcDraft = null;
  calibrationPanel.hidden = true;
  setActiveFieldTool("select");
  updatePitchSetupWorkflow();
  drawOverlay();
}

function ensureCurrentCalibration() {
  let calibration = calibrationForCurrentFrame();
  if (calibration) return calibration;
  const initialProjection = existingProjectionForCurrentFrame();
  calibration = {
    id: `calibration-${Date.now()}`,
    video: selectedVideo.name,
    frame: currentFrameNumber(),
    source: "manual",
    correspondences: [],
    constraints: [],
    homography: null,
    reprojectionError: null,
  };
  if (initialProjection) {
    calibration.initialHomography = JSON.parse(JSON.stringify(initialProjection.homography));
    calibration.initialHomographySource = initialProjection.source;
  }
  fieldCalibrations.push(calibration);
  return calibration;
}

function calibrationHitTest(coordinates) {
  const calibration = calibrationForCurrentFrame();
  if (!calibration || !displayedVideoRect) return null;
  const hitRadius = 14 / (displayedVideoRect.width / video.videoWidth);
  const constraint = calibrationConstraints(calibration).find((item) => {
    if (item.type === "point" && distanceBetween(item.image, coordinates) <= hitRadius) return { item, handle: "point" };
    if (item.type === "line") {
      if (distanceBetween(item.image.start, coordinates) <= hitRadius) return { item, handle: "start" };
      if (distanceBetween(item.image.end, coordinates) <= hitRadius) return { item, handle: "end" };
      if (distanceToLineSegment(coordinates, item.image.start, item.image.end) <= hitRadius) return { item, handle: "line" };
    }
    if (item.type === "arc") {
      const { center, radiusX, radiusY, rotation } = item.image.geometry;
      const xHandle = { x: center.x + radiusX * Math.cos(rotation), y: center.y + radiusX * Math.sin(rotation) };
      const yHandle = { x: center.x - radiusY * Math.sin(rotation), y: center.y + radiusY * Math.cos(rotation) };
      if (distanceBetween(coordinates, center) <= hitRadius) return { item, handle: "arc-center" };
      if (distanceBetween(coordinates, xHandle) <= hitRadius) return { item, handle: "arc-radius-x" };
      if (distanceBetween(coordinates, yHandle) <= hitRadius) return { item, handle: "arc-radius-y" };
      const relativeX = coordinates.x - center.x;
      const relativeY = coordinates.y - center.y;
      const localX = relativeX * Math.cos(rotation) + relativeY * Math.sin(rotation);
      const localY = -relativeX * Math.sin(rotation) + relativeY * Math.cos(rotation);
      if (Math.abs(Math.hypot(localX / radiusX, localY / radiusY) - 1) * Math.min(radiusX, radiusY) <= hitRadius) return { item, handle: "arc" };
    }
    return null;
  });
  return constraint ? { calibration, constraint: constraint.item, handle: constraint.handle } : null;
}

function isCalibrationHandle(hit) {
  return hit && hit.handle !== "line" && hit.handle !== "arc";
}

function beginCalibrationConstraintDrag(hit, coordinates, pointerId) {
  setCalibrationSelection(hit.constraint.type, hit.constraint.semantic);
  selectedCalibrationConstraint = {
    calibrationId: hit.calibration.id,
    type: hit.constraint.type,
    semantic: hit.constraint.semantic,
  };
  calibrationDragState = {
    ...hit,
    origin: coordinates,
    image: JSON.parse(JSON.stringify(hit.constraint.image)),
    before: cloneFieldState(),
    didMove: false,
  };
  workArea.setPointerCapture(pointerId);
  updateCurrentReferenceLabel();
  if (calibrationMode) updateCalibrationStatus();
}

async function solveCalibration(calibration) {
  cameraAtlasRequestSequence += 1;
  cameraAtlasResult = null;
  anchorSelectionResult = null;
  anchorSelectionError = null;
  invalidatePropagationsFromAnchor(calibration);
  invalidateRefinedSegmentsForCalibration(calibration);
  const constraints = calibrationConstraints(calibration);
  const pointCount = constraints.filter((item) => item.type === "point").length;
  const isRefinement = hasInitialHomography(calibration);
  const sequence = ++calibrationSolveSequence;
  calibrationSolveError = null;
  if (!constraints.length || (!isRefinement && pointCount < 4 && !constraints.some((item) => item.type !== "point"))) {
    calibrationSolvePending = null;
    calibration.homography = null;
    calibration.reprojectionError = null;
    updateCalibrationStatus();
    drawOverlay();
    return;
  }
  calibrationSolvePending = { calibrationId: calibration.id, sequence };
  updateCalibrationStatus();
  try {
    const response = await fetch("/api/field-calibration/solve", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        constraints,
        image_width: video.videoWidth,
        image_height: video.videoHeight,
        initial_homography: calibration.initialHomography || null,
      }),
    });
    const result = await response.json();
    const solved = result.status === "solved" || result.status === "solved_with_warnings";
    if (!response.ok || !solved) throw new Error(result.detail || "No se pudo calcular la proyección del campo");
    if (sequence !== calibrationSolveSequence || !fieldCalibrations.includes(calibration)) return;
    calibrationSolvePending = null;
    calibration.homography = result.homography;
    calibration.reprojectionError = result.reprojection_error;
    calibration.camera = result.camera || null;
    calibration.diagnostics = result.diagnostics || null;
    updateCalibrationStatus();
    updatePitchSetupWorkflow();
    drawOverlay();
  } catch (error) {
    if (sequence !== calibrationSolveSequence || !fieldCalibrations.includes(calibration)) return;
    calibrationSolvePending = null;
    calibrationSolveError = {
      calibrationId: calibration.id,
      message: error instanceof Error ? error.message : "No se pudo calcular la proyección del campo.",
    };
    updateCalibrationStatus();
    updatePitchSetupWorkflow();
    drawOverlay();
  }
}

function addCalibrationCorrespondence(image) {
  const landmark = PITCH_MODEL.landmarksById[calibrationSemantic];
  if (!landmark) return;
  rememberFieldEdit();
  const calibration = ensureCurrentCalibration();
  const existing = calibrationConstraints(calibration).find((item) => item.type === "point" && item.semantic === landmark.id);
  const constraint = { type: "point", semantic: landmark.id, world: { ...landmark.world }, image: clampToVideo(image), source: "manual" };
  if (existing) Object.assign(existing, constraint);
  else calibration.constraints.push(constraint);
  syncLegacyPointCorrespondences(calibration);
  solveCalibration(calibration);
}

function removeSelectedCalibrationCorrespondence() {
  const calibration = calibrationForCurrentFrame();
  if (!calibration) return;
  const landmarkId = calibrationSemantic;
  if (!calibrationConstraints(calibration).some((item) => item.type === calibrationType && item.semantic === landmarkId)) return;
  rememberFieldEdit();
  calibration.constraints = calibrationConstraints(calibration).filter((item) => !(item.type === calibrationType && item.semantic === landmarkId));
  syncLegacyPointCorrespondences(calibration);
  solveCalibration(calibration);
}

function clearCurrentCalibration() {
  const calibration = calibrationForCurrentFrame();
  if (!calibration) return;
  cameraAtlasRequestSequence += 1;
  cameraAtlasResult = null;
  anchorSelectionResult = null;
  anchorSelectionError = null;
  rememberFieldEdit();
  invalidatePropagationsFromAnchor(calibration);
  invalidateRefinedSegmentsForCalibration(calibration);
  fieldCalibrations = fieldCalibrations.filter((item) => item.id !== calibration.id);
  calibrationSolveSequence += 1;
  calibrationSolvePending = null;
  calibrationSolveError = null;
  updateCalibrationStatus();
  drawOverlay();
}

function drawFieldHandle({ x, y }, selected, displayScale) {
  const radius = (selected ? 8 : 6) / displayScale;
  overlayContext.beginPath();
  overlayContext.arc(x, y, radius, 0, Math.PI * 2);
  overlayContext.fillStyle = selected ? "#f0f2f4" : "#2ec4b6";
  overlayContext.fill();
  overlayContext.strokeStyle = "#0b3532";
  overlayContext.lineWidth = 2 / displayScale;
  overlayContext.stroke();
}

function drawEllipse(annotation, selected, displayScale) {
  const { center, radiusX, radiusY, rotation, arc } = annotation.coordinates;
  const arcSpan = arc ? Math.abs(arc.endAngle - arc.startAngle) : Math.PI * 2;
  const isPartialArc = arcSpan < Math.PI * 1.85;
  overlayContext.strokeStyle = selected ? "#f0f2f4" : "#2ec4b6";
  overlayContext.lineWidth = (selected ? 3 : 2.5) / displayScale;
  overlayContext.beginPath();
  overlayContext.ellipse(
    center.x,
    center.y,
    radiusX,
    radiusY,
    rotation,
    isPartialArc ? arc.startAngle : 0,
    isPartialArc ? arc.endAngle : Math.PI * 2,
    isPartialArc ? arc.counterclockwise : false,
  );
  overlayContext.stroke();

  if (selected) {
    const cos = Math.cos(rotation);
    const sin = Math.sin(rotation);
    overlayContext.setLineDash([5 / displayScale, 4 / displayScale]);
    overlayContext.beginPath();
    overlayContext.moveTo(center.x, center.y);
    overlayContext.lineTo(center.x + radiusX * cos, center.y + radiusX * sin);
    overlayContext.moveTo(center.x, center.y);
    overlayContext.lineTo(center.x - radiusY * sin, center.y + radiusY * cos);
    overlayContext.stroke();
    overlayContext.setLineDash([]);
    drawFieldHandle(center, true, displayScale);
    drawFieldHandle({ x: center.x + radiusX * cos, y: center.y + radiusX * sin }, true, displayScale);
    drawFieldHandle({ x: center.x - radiusY * sin, y: center.y + radiusY * cos }, true, displayScale);
  }
}

function drawRawAssistedStroke(displayScale) {
  if (!assistedDraft || assistedDraft.samples.length < 2) return;
  overlayContext.save();
  overlayContext.strokeStyle = "rgba(240, 242, 244, .55)";
  overlayContext.lineWidth = 2 / displayScale;
  overlayContext.setLineDash([6 / displayScale, 5 / displayScale]);
  overlayContext.beginPath();
  assistedDraft.samples.forEach((point, index) => {
    if (index === 0) overlayContext.moveTo(point.x, point.y);
    else overlayContext.lineTo(point.x, point.y);
  });
  overlayContext.stroke();
  overlayContext.restore();
}

const PROJECTIVE_DENOMINATOR_EPSILON = 1e-8;

function projectWorldPointDetailed(point, homography) {
  const [first, second, third] = homography;
  const denominator = third[0] * point.x + third[1] * point.y + third[2];
  if (!Number.isFinite(denominator) || Math.abs(denominator) < PROJECTIVE_DENOMINATOR_EPSILON) return null;
  const x = (first[0] * point.x + first[1] * point.y + first[2]) / denominator;
  const y = (second[0] * point.x + second[1] * point.y + second[2]) / denominator;
  if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
  return {
    x,
    y,
    denominator,
  };
}

function projectWorldPoint(point, homography) {
  const projected = projectWorldPointDetailed(point, homography);
  return projected ? { x: projected.x, y: projected.y } : null;
}

function sampleWorldPolyline(points, samplesPerSegment = 96) {
  const samples = [];
  for (let segmentIndex = 1; segmentIndex < points.length; segmentIndex += 1) {
    const start = points[segmentIndex - 1];
    const end = points[segmentIndex];
    for (let sampleIndex = 0; sampleIndex <= samplesPerSegment; sampleIndex += 1) {
      if (segmentIndex > 1 && sampleIndex === 0) continue;
      const ratio = sampleIndex / samplesPerSegment;
      samples.push({
        x: start.x + (end.x - start.x) * ratio,
        y: start.y + (end.y - start.y) * ratio,
      });
    }
  }
  return samples;
}

function drawProjectedPath(points, homography, displayScale) {
  const maximumJump = Math.hypot(video.videoWidth, video.videoHeight) * 2;
  let previous = null;
  let started = false;
  let hasPath = false;
  overlayContext.beginPath();
  points.forEach((point) => {
    const projected = projectWorldPointDetailed(point, homography);
    if (!projected) {
      previous = null;
      started = false;
      return;
    }
    const denominatorChangedSign = previous && Math.sign(previous.denominator) !== Math.sign(projected.denominator);
    const jump = previous ? Math.hypot(projected.x - previous.x, projected.y - previous.y) : 0;
    const discontinuity = !previous || denominatorChangedSign || jump > maximumJump;
    if (discontinuity || !started) {
      overlayContext.moveTo(projected.x, projected.y);
      started = true;
    } else {
      overlayContext.lineTo(projected.x, projected.y);
    }
    hasPath = true;
    previous = projected;
  });
  if (hasPath) overlayContext.stroke();
}

function drawProjectedPitch(homography, displayScale, style = {}) {
  const color = style.color || "#5aa9ff";
  overlayContext.strokeStyle = color;
  overlayContext.lineWidth = style.lineWidth || 2 / displayScale;
  overlayContext.setLineDash(style.lineDash || []);
  PITCH_MODEL.segments.forEach((segment) => {
    drawProjectedPath(sampleWorldPolyline(segment), homography, displayScale);
  });
  PITCH_MODEL.calibrationArcs.forEach((arc) => {
    const { center, radius, startAngle, endAngle } = arc.world;
    const points = Number.isFinite(startAngle) && Number.isFinite(endAngle)
      ? PITCH_MODEL.sampleArc(center, radius, startAngle, endAngle, 96)
      : PITCH_MODEL.sampleCircle(center, radius, 128);
    drawProjectedPath(points, homography, displayScale);
  });
  PITCH_MODEL.markPoints.forEach((worldPoint) => {
    const point = projectWorldPoint(worldPoint, homography);
    if (!point) return;
    overlayContext.beginPath();
    overlayContext.arc(point.x, point.y, 3.5 / displayScale, 0, Math.PI * 2);
    overlayContext.fillStyle = color;
    overlayContext.fill();
  });
  overlayContext.setLineDash([]);
}

function drawCalibrationHandle({ x, y }, displayScale) {
  overlayContext.beginPath();
  overlayContext.arc(x, y, 7 / displayScale, 0, Math.PI * 2);
  overlayContext.fillStyle = "#d4a414";
  overlayContext.fill();
  overlayContext.strokeStyle = "#fff0a8";
  overlayContext.lineWidth = 2 / displayScale;
  overlayContext.stroke();
}

function drawCalibrationArc(geometry, selected, displayScale) {
  const { center, radiusX, radiusY, rotation, arc } = geometry;
  const arcSpan = arc ? Math.abs(arc.endAngle - arc.startAngle) : Math.PI * 2;
  const isPartialArc = arcSpan < Math.PI * 1.85;
  overlayContext.strokeStyle = selected ? "#d4a414" : "rgba(212, 164, 20, .72)";
  overlayContext.lineWidth = (selected ? 3 : 2) / displayScale;
  overlayContext.beginPath();
  overlayContext.ellipse(
    center.x,
    center.y,
    radiusX,
    radiusY,
    rotation,
    isPartialArc ? arc.startAngle : 0,
    isPartialArc ? arc.endAngle : Math.PI * 2,
    arc?.counterclockwise || false,
  );
  overlayContext.stroke();

  if (!selected) return;
  const cos = Math.cos(rotation);
  const sin = Math.sin(rotation);
  drawCalibrationHandle(center, displayScale);
  drawCalibrationHandle({ x: center.x + radiusX * cos, y: center.y + radiusX * sin }, displayScale);
  drawCalibrationHandle({ x: center.x - radiusY * sin, y: center.y + radiusY * cos }, displayScale);
}

function drawCalibrationOverlay() {
  const calibration = calibrationForCurrentFrame();
  const refinedFrame = refinedFrameForCurrentFrame();
  const propagatedFrame = propagatedFrameForCurrentFrame();
  if (!displayedVideoRect || !video.videoWidth) return;
  const displayScale = displayedVideoRect.width / video.videoWidth;
  overlayContext.save();

  const homography = hasSolvedManualCalibration(calibration)
    ? calibration.homography
    : refinedFrame?.homography || propagatedFrame?.homography;
  if (Array.isArray(homography)) drawProjectedPitch(homography, displayScale);

  if (!calibration) {
    overlayContext.restore();
    return;
  }

  const lineWarnings = new Map(
    calibrationLineDiagnostics(calibration)
      .filter((diagnostic) => (
        diagnostic.quality === "bad" || (!diagnostic.quality && diagnostic.errorPx > LINE_WARNING_ERROR_PX)
      ))
      .map((diagnostic) => [diagnostic.semantic, diagnostic])
  );
  calibrationConstraints(calibration).forEach((constraint, index) => {
    const selected = (
      (calibrationMode && constraint.semantic === calibrationSemantic && constraint.type === calibrationType)
      || (
        selectedCalibrationConstraint?.calibrationId === calibration.id
        && selectedCalibrationConstraint?.semantic === constraint.semantic
        && selectedCalibrationConstraint?.type === constraint.type
      )
    );
    const hasWarning = constraint.type === "line" && lineWarnings.has(constraint.semantic);
    if (!calibrationMode && !selected && !debugOverlayEnabled && !hasWarning) return;
    const radius = (selected ? 9 : 7) / displayScale;
    const constraintColor = selected || hasWarning ? "#d4a414" : "rgba(212, 164, 20, .52)";
    if (constraint.type === "line") {
      overlayContext.strokeStyle = constraintColor;
      overlayContext.lineWidth = (selected || hasWarning ? 3 : 1.75) / displayScale;
      overlayContext.beginPath();
      overlayContext.moveTo(constraint.image.start.x, constraint.image.start.y);
      overlayContext.lineTo(constraint.image.end.x, constraint.image.end.y);
      overlayContext.stroke();
      if (selected) {
        drawCalibrationHandle(constraint.image.start, displayScale);
        drawCalibrationHandle(constraint.image.end, displayScale);
      }
      const midpoint = { x: (constraint.image.start.x + constraint.image.end.x) / 2, y: (constraint.image.start.y + constraint.image.end.y) / 2 };
      overlayContext.fillStyle = constraintColor;
      overlayContext.fillText(String(index + 1), midpoint.x, midpoint.y);
      return;
    }
    if (constraint.type === "arc") {
      drawCalibrationArc(constraint.image.geometry, selected, displayScale);
      return;
    }
    overlayContext.beginPath();
    overlayContext.arc(constraint.image.x, constraint.image.y, radius, 0, Math.PI * 2);
    overlayContext.fillStyle = constraintColor;
    overlayContext.fill();
    overlayContext.strokeStyle = selected ? "#fff0a8" : "#755c0c";
    overlayContext.lineWidth = 2 / displayScale;
    overlayContext.stroke();
    overlayContext.fillStyle = "#17191c";
    overlayContext.font = `700 ${11 / displayScale}px ui-monospace, SFMono-Regular, Menlo, monospace`;
    overlayContext.textAlign = "center";
    overlayContext.textBaseline = "middle";
    overlayContext.fillText(String(index + 1), constraint.image.x, constraint.image.y);
  });
  if (calibrationLineDraft) {
    overlayContext.setLineDash([7 / displayScale, 5 / displayScale]);
    overlayContext.strokeStyle = "#d4a414";
    overlayContext.lineWidth = 2.5 / displayScale;
    overlayContext.beginPath();
    overlayContext.moveTo(calibrationLineDraft.start.x, calibrationLineDraft.start.y);
    overlayContext.lineTo(calibrationLineDraft.end.x, calibrationLineDraft.end.y);
    overlayContext.stroke();
    overlayContext.setLineDash([]);
  }
  if (calibrationArcDraft?.samples.length > 1) {
    overlayContext.setLineDash([6 / displayScale, 5 / displayScale]);
    overlayContext.strokeStyle = "#d4a414";
    overlayContext.lineWidth = 2 / displayScale;
    overlayContext.beginPath();
    calibrationArcDraft.samples.forEach((point, index) => {
      if (index === 0) overlayContext.moveTo(point.x, point.y);
      else overlayContext.lineTo(point.x, point.y);
    });
    overlayContext.stroke();
    overlayContext.setLineDash([]);
  }
  overlayContext.restore();
}

function drawFieldAnnotations() {
  if (!displayedVideoRect || !video.videoWidth || !video.videoHeight) return;
  const displayScale = displayedVideoRect.width / video.videoWidth;
  const selectedId = selectedFieldItem?.id;
  overlayContext.save();
  overlayContext.lineCap = "round";
  overlayContext.lineJoin = "round";

  annotationsForCurrentFrame().forEach((annotation) => {
    const selected = annotation.id === selectedId;
    const color = selected ? "#f0f2f4" : "#2ec4b6";
    if (annotation.type === "point") {
      drawFieldHandle(annotation.coordinates, selected, displayScale);
      return;
    }

    if (annotation.type === "ellipse") {
      drawEllipse(annotation, selected, displayScale);
      return;
    }

    const { start, end } = annotation.coordinates;
    overlayContext.strokeStyle = color;
    overlayContext.lineWidth = (selected ? 3 : 2.5) / displayScale;
    overlayContext.beginPath();
    overlayContext.moveTo(start.x, start.y);
    overlayContext.lineTo(end.x, end.y);
    overlayContext.stroke();
    if (selected) {
      drawFieldHandle(start, true, displayScale);
      drawFieldHandle(end, true, displayScale);
    }
  });

  if (lineDraft) {
    overlayContext.setLineDash([7 / displayScale, 5 / displayScale]);
    overlayContext.strokeStyle = "#2ec4b6";
    overlayContext.lineWidth = 2.5 / displayScale;
    overlayContext.beginPath();
    overlayContext.moveTo(lineDraft.start.x, lineDraft.start.y);
    overlayContext.lineTo(lineDraft.end.x, lineDraft.end.y);
    overlayContext.stroke();
    overlayContext.setLineDash([]);
    drawFieldHandle(lineDraft.start, true, displayScale);
    drawFieldHandle(lineDraft.end, true, displayScale);
  }
  drawRawAssistedStroke(displayScale);
  overlayContext.restore();
}

function drawDebugOverlay() {
  if (!debugOverlayEnabled || !displayedVideoRect || !video.videoWidth || !video.videoHeight) return;

  const displayScale = displayedVideoRect.width / video.videoWidth;
  const markerRadius = 6 / displayScale;
  const padding = 16 / displayScale;
  const fontSize = 12 / displayScale;
  const crosshairSize = 12 / displayScale;

  overlayContext.save();
  overlayContext.strokeStyle = "#d4a414";
  overlayContext.fillStyle = "#f0f2f4";
  overlayContext.lineWidth = 2 / displayScale;
  overlayContext.font = `700 ${fontSize}px ui-monospace, SFMono-Regular, Menlo, monospace`;
  overlayContext.textBaseline = "top";

  [[0, 0], [video.videoWidth, 0], [video.videoWidth, video.videoHeight], [0, video.videoHeight]].forEach(([x, y]) => {
    overlayContext.beginPath();
    overlayContext.arc(x, y, markerRadius, 0, Math.PI * 2);
    overlayContext.fill();
    overlayContext.stroke();
  });

  const centerX = video.videoWidth / 2;
  const centerY = video.videoHeight / 2;
  overlayContext.beginPath();
  overlayContext.moveTo(centerX - crosshairSize, centerY);
  overlayContext.lineTo(centerX + crosshairSize, centerY);
  overlayContext.moveTo(centerX, centerY - crosshairSize);
  overlayContext.lineTo(centerX, centerY + crosshairSize);
  overlayContext.stroke();

  overlayContext.fillText(`${video.videoWidth} × ${video.videoHeight}`, padding, padding);

  const strategyLabel = calibrationStrategyLabel(calibrationForCurrentFrame());
  if (strategyLabel) {
    overlayContext.fillStyle = "#d9c4ff";
    overlayContext.fillText(strategyLabel, padding, padding + 60 / displayScale);
    overlayContext.fillStyle = "#f0f2f4";
  }

  if (pointerVideoCoordinates) {
    const x = Math.round(pointerVideoCoordinates.x);
    const y = Math.round(pointerVideoCoordinates.y);
    const label = `${x}, ${y}`;
    const labelX = Math.min(pointerVideoCoordinates.x + padding, video.videoWidth - 100 / displayScale);
    const labelY = Math.min(pointerVideoCoordinates.y + padding, video.videoHeight - 22 / displayScale);
    overlayContext.fillText(label, labelX, labelY);
  }
  overlayContext.restore();
}

function drawOverlay() {
  overlayContext.clearRect(0, 0, overlay.width, overlay.height);
  if (activeMode === "guided" && calibrationMode) {
    drawFieldAnnotations();
    drawCalibrationOverlay();
    if (debugOverlayEnabled && calibrationMode) drawDebugOverlay();
    updateCurrentReferenceLabel();
  }
  updatePitchSetupWorkflow();
}

function syncOverlay() {
  displayedVideoRect = getDisplayedVideoRect();
  if (!displayedVideoRect) {
    overlay.classList.remove("is-loaded");
    return;
  }

  const workspaceBounds = workArea.getBoundingClientRect();
  overlay.width = video.videoWidth;
  overlay.height = video.videoHeight;
  overlay.style.left = `${displayedVideoRect.left - workspaceBounds.left}px`;
  overlay.style.top = `${displayedVideoRect.top - workspaceBounds.top}px`;
  overlay.style.width = `${displayedVideoRect.width}px`;
  overlay.style.height = `${displayedVideoRect.height}px`;
  overlay.classList.add("is-loaded");
  pointerVideoCoordinates = pointerBrowserPosition ? browserToVideoCoordinates(pointerBrowserPosition) : null;
  clampVisibleCalibrationPanel();
  drawOverlay();
}

function formatTime(seconds) {
  const totalSeconds = Math.max(0, Math.floor(seconds || 0));
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const remainingSeconds = totalSeconds % 60;
  return [hours, minutes, remainingSeconds].map((value) => String(value).padStart(2, "0")).join(":");
}

function setControlsEnabled(enabled) {
  playPauseButton.disabled = !enabled;
  timeline.disabled = !enabled;
  previousFrameButtons.forEach((button) => { button.disabled = !enabled; });
  nextFrameButtons.forEach((button) => { button.disabled = !enabled; });
}

function updatePlaybackDisplay() {
  const duration = Number.isFinite(video.duration) ? video.duration : (selectedVideo?.duration || 0);
  const currentTime = video.currentTime || 0;
  const fps = selectedVideo?.fps;
  timeline.max = duration;
  timeline.value = Math.min(currentTime, duration);
  currentTimeLabel.textContent = formatTime(currentTime);
  totalTimeLabel.textContent = formatTime(duration);

  if (fps) {
    const currentFrame = Math.min(Math.floor(currentTime * fps), Math.max(0, (selectedVideo.frame_count || Infinity) - 1));
    const totalFrames = selectedVideo.frame_count ?? "—";
    frameReadout.textContent = `Fotograma ${currentFrame} / ${totalFrames}`;
  } else {
    frameReadout.textContent = "Fotograma —";
  }
}

function resetPlaybackState() {
  video.pause();
  video.currentTime = 0;
  playPauseButton.textContent = "▶";
  playPauseButton.setAttribute("aria-label", "Reproducir");
  timeline.max = 0;
  timeline.value = 0;
  currentTimeLabel.textContent = "00:00:00";
  totalTimeLabel.textContent = "00:00:00";
  frameReadout.textContent = "Fotograma —";
  pointerVideoCoordinates = null;
  pointerBrowserPosition = null;
  displayedVideoRect = null;
  overlayContext.clearRect(0, 0, overlay.width, overlay.height);
  overlay.classList.remove("is-loaded");
}

async function loadVideo(videoInfo) {
  const loadSequence = ++videoLoadSequence;
  cameraAtlasRequestSequence += 1;
  cameraAtlasResult = null;
  anchorSelectionResult = null;
  anchorSelectionError = null;
  anchorSelectionInProgress = false;
  viewScanWorkflow = null;
  selectedVideo = videoInfo;
  fieldAnnotations = [];
  fieldCalibrations = [];
  fieldPropagations = [];
  fieldRefinedSegments = [];
  fieldPitchSetup = null;
  applyPitchModel(null);
  fieldUndoStack = [];
  selectedFieldItem = null;
  selectedCalibrationConstraint = null;
  lineDraft = null;
  assistedDraft = null;
  setCalibrationMode(false);
  calibrationPanel.hidden = true;
  calibrationDragState = null;
  calibrationLineDraft = null;
  calibrationArcDraft = null;
  calibrationSolvePending = null;
  calibrationSolveError = null;
  setReviewDirty(false);
  resetPlaybackState();
  setControlsEnabled(false);
  video.classList.remove("is-loaded");
  workspaceMessage.hidden = false;
  workspaceTitle.textContent = "Cargando vídeo";
  workspaceDescription.textContent = videoInfo.name;
  video.src = `/videos/${encodeURIComponent(videoInfo.name)}`;
  video.load();

  try {
    const response = await fetch(`/api/reviews/${encodeURIComponent(videoInfo.name)}`);
    if (!response.ok) throw new Error("No se pudo cargar la revisión");
    const { review } = await response.json();
    if (loadSequence !== videoLoadSequence) return;
    fieldAnnotations = Array.isArray(review?.field?.annotations) ? review.field.annotations : [];
    fieldCalibrations = Array.isArray(review?.field?.calibrations) ? review.field.calibrations : [];
    fieldPropagations = Array.isArray(review?.field?.propagations) ? review.field.propagations : [];
    fieldRefinedSegments = Array.isArray(review?.field?.refinedSegments) ? review.field.refinedSegments : [];
    fieldPitchSetup = review?.field?.pitch_setup && typeof review.field.pitch_setup === "object"
      ? review.field.pitch_setup
      : null;
    applyPitchModel(review?.field?.pitch_model);
    fieldCalibrations.forEach((calibration) => calibrationConstraints(calibration));
    annotationSequence = fieldAnnotations.length;
    setReviewDirty(false);
    updatePitchSetupWorkflow();
    drawOverlay();
  } catch (error) {
    if (loadSequence !== videoLoadSequence) return;
    setReviewDirty(true);
    workspaceDescription.textContent = "No se pudo cargar la revisión. Puedes crear una y guardarla.";
  }
}

async function saveCurrentReview() {
  if (!selectedVideo) return false;
  saveButton.disabled = true;
  const review = {
    format_version: 1,
    video: { filename: selectedVideo.name },
    field: {
      annotations: fieldAnnotations.filter((annotation) => annotation.video === selectedVideo.name),
      calibrations: fieldCalibrations.filter((calibration) => (
        calibration.video === selectedVideo.name && calibrationConstraints(calibration).length > 0
      )),
      propagations: fieldPropagations.filter((propagation) => propagation.video === selectedVideo.name),
      refinedSegments: fieldRefinedSegments.filter((segment) => segment.video === selectedVideo.name),
      pitch_model: { ...fieldPitchModel },
      pitch_setup: fieldPitchSetup?.video === selectedVideo.name ? fieldPitchSetup : null,
    },
    people: {},
  };

  try {
    const response = await fetch(`/api/reviews/${encodeURIComponent(selectedVideo.name)}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(review),
    });
    if (!response.ok) throw new Error("No se pudo guardar la revisión");
    setReviewDirty(false);
    return true;
  } catch (error) {
    setReviewDirty(true);
    saveStatusLabel.textContent = "ERROR AL GUARDAR";
    return false;
  } finally {
    saveButton.disabled = false;
  }
}

async function propagateCurrentCalibration() {
  const calibration = calibrationForCurrentFrame();
  if (!selectedVideo || !hasSolvedManualCalibration(calibration)) return;

  video.pause();
  const requestSequence = ++propagationRequestSequence;
  const solverSequence = calibrationSolveSequence;
  const anchorFrame = currentFrameNumber();
  const filename = selectedVideo.name;
  const homography = JSON.parse(JSON.stringify(calibration.homography));
  const frameCount = knownVideoFrameCount();
  const nextAnchor = nextManualCalibration(calibration);
  if (!frameCount) {
    setPropagationStatus("No se conoce el total de fotogramas del vídeo.");
    return;
  }
  const horizonEnd = Math.min(frameCount, nextAnchor?.frame ?? frameCount);
  const maxFrames = Math.max(0, horizonEnd - anchorFrame - 1);
  if (!maxFrames) {
    setPropagationStatus(nextAnchor
      ? `La siguiente ancla manual está en el fotograma ${nextAnchor.frame}.`
      : "La calibración manual ya está en el último fotograma.");
    return;
  }
  propagationInProgress = true;
  setPropagationStatus("Propagando FIELD…");
  updatePropagationControl();

  try {
    const response = await fetch("/api/field-propagation", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        filename,
        anchor_frame: anchorFrame,
        homography,
        max_frames: maxFrames,
        direction: 1,
      }),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || "No se pudo propagar el campo");
    if (
      requestSequence !== propagationRequestSequence
      || solverSequence !== calibrationSolveSequence
      || selectedVideo?.name !== filename
    ) return;
    if (result.status === "invalid") throw new Error(result.detail || "No se pudo propagar el campo");

    const frames = Array.isArray(result.frames)
      ? result.frames.filter((item) => item.frame !== anchorFrame && Array.isArray(item.homography))
      : [];
    rememberFieldEdit();
    fieldPropagations = fieldPropagations.filter((propagation) => !(
      propagation.video === filename
      && propagation.anchorFrame === anchorFrame
      && propagation.direction === "forward"
    ));
    if (frames.length) {
      fieldPropagations.push({
        id: `propagation-${Date.now()}`,
        video: filename,
        anchorFrame,
        direction: "forward",
        source: "propagated_optical_flow",
        status: result.status,
        detail: result.detail,
        stoppedFrame: result.stopped_frame,
        requestedFrames: result.requested_frames,
        propagatedFrames: result.propagated_frames,
        fps: result.fps,
        config: result.config,
        frames,
      });
    }
    invalidateAutoRefinementPreview();

    if (result.status === "ok") {
      if (horizonEnd === frameCount) {
        setPropagationStatus("FIELD propagado hasta el final del vídeo.");
      } else {
        setPropagationStatus(`FIELD propagado hasta la siguiente ancla manual (fotograma ${nextAnchor.frame}).`);
      }
    } else {
      const reviewFrame = result.stopped_frame ?? frames[frames.length - 1]?.frame ?? anchorFrame;
      setPropagationStatus(`Propagado hasta el fotograma ${reviewFrame}. Se necesita revisión: ${result.detail || "confianza insuficiente."}`);
    }
    drawOverlay();
  } catch (error) {
    if (requestSequence === propagationRequestSequence) {
      setPropagationStatus(error.message || "No se pudo propagar el campo.");
    }
  } finally {
    if (requestSequence === propagationRequestSequence) {
      propagationInProgress = false;
      updatePropagationControl();
    }
  }
}

function formatFrameRanges(ranges) {
  return (ranges || [])
    .map(({ start, end }) => (start === end ? String(start) : `${start}–${end}`))
    .join(", ");
}

async function refinePreviousSegment() {
  const rightAnchor = calibrationForCurrentFrame();
  const leftAnchor = previousManualCalibration(rightAnchor);
  if (
    !selectedVideo
    || isFieldEditInProgress()
    || !hasSolvedManualCalibration(rightAnchor)
    || !leftAnchor
  ) return;

  video.pause();
  const requestSequence = ++refinementRequestSequence;
  const solverSequence = calibrationSolveSequence;
  const calibrationRevision = manualCalibrationRevision;
  const filename = selectedVideo.name;
  const leftAnchorFrame = leftAnchor.frame;
  const rightAnchorFrame = rightAnchor.frame;
  refinementInProgress = true;
  setPropagationStatus("Refinando tramo FIELD…");
  updatePropagationControl();
  updateRefinementControl();

  try {
    const response = await fetch("/api/field-segment/refine", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        filename,
        left_anchor_frame: leftAnchorFrame,
        left_homography: leftAnchor.homography,
        right_anchor_frame: rightAnchorFrame,
        right_homography: rightAnchor.homography,
      }),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || "No se pudo refinar el tramo FIELD");
    if (
      requestSequence !== refinementRequestSequence
      || solverSequence !== calibrationSolveSequence
      || calibrationRevision !== manualCalibrationRevision
      || selectedVideo?.name !== filename
    ) return;
    if (result.status === "invalid") throw new Error(result.detail || "No se pudo refinar el tramo FIELD");

    rememberFieldEdit();
    fieldRefinedSegments = fieldRefinedSegments.filter((segment) => !(
      segment.video === filename
      && segment.leftAnchorFrame === leftAnchorFrame
      && segment.rightAnchorFrame === rightAnchorFrame
    ));
    fieldRefinedSegments.push({
      id: `refined-segment-${Date.now()}`,
      video: filename,
      leftAnchorFrame,
      rightAnchorFrame,
      source: "refined_bidirectional",
      status: result.status,
      detail: result.detail,
      intervalFrames: result.interval_frames,
      producedFrames: result.produced_frames,
      reviewRanges: result.review_ranges || [],
      missingRanges: result.missing_ranges || [],
      forward: result.forward,
      backward: result.backward,
      config: result.config,
      frames: result.frames || [],
    });
    invalidateAutoRefinementPreview();

    const missingRanges = formatFrameRanges(result.missing_ranges);
    const reviewRanges = formatFrameRanges(result.review_ranges);
    if (missingRanges) {
      setPropagationStatus(`Refinado parcial. Sin cobertura en ${missingRanges}.`);
    } else if (reviewRanges) {
      setPropagationStatus(`Tramo refinado. Revisar fotogramas ${reviewRanges}.`);
    } else if (result.status === "partial") {
      setPropagationStatus(`Refinado parcial: ${result.detail || "se necesita revisión."}`);
    } else {
      setPropagationStatus(`Tramo refinado: fotogramas ${leftAnchorFrame}–${rightAnchorFrame}.`);
    }
    drawOverlay();
  } catch (error) {
    if (requestSequence === refinementRequestSequence) {
      setPropagationStatus(error.message || "No se pudo refinar el tramo FIELD.");
    }
  } finally {
    if (requestSequence === refinementRequestSequence) {
      refinementInProgress = false;
      updatePropagationControl();
      updateRefinementControl();
    }
  }
}

function stepFrame(direction) {
  if (!selectedVideo?.fps || !Number.isFinite(video.duration)) return;
  video.pause();
  const frameDuration = 1 / selectedVideo.fps;
  const targetTime = Math.min(Math.max(video.currentTime + direction * frameDuration, 0), Math.max(video.duration - frameDuration, 0));
  video.currentTime = targetTime;
}

playPauseButton.addEventListener("click", () => {
  if (video.paused) video.play();
  else video.pause();
});

previousFrameButtons.forEach((button) => button.addEventListener("click", () => stepFrame(-1)));
nextFrameButtons.forEach((button) => button.addEventListener("click", () => stepFrame(1)));

timeline.addEventListener("input", () => {
  video.currentTime = Number(timeline.value);
});

video.addEventListener("loadedmetadata", () => {
  video.currentTime = 0;
  workspaceMessage.hidden = true;
  video.classList.add("is-loaded");
  setControlsEnabled(true);
  updatePlaybackDisplay();
  syncOverlay();
  updatePitchSetupWorkflow();
});

video.addEventListener("timeupdate", () => {
  if (viewScanWorkflow?.contextEndTime && video.currentTime >= viewScanWorkflow.contextEndTime) {
    video.pause();
    viewScanWorkflow.contextEndTime = null;
  }
  updatePlaybackDisplay();
  drawOverlay();
});
video.addEventListener("seeked", () => {
  updatePlaybackDisplay();
  if (calibrationMode) updateCalibrationStatus();
  drawOverlay();
});
video.addEventListener("play", () => {
  playPauseButton.textContent = "❚❚";
  playPauseButton.setAttribute("aria-label", "Pausar");
  drawOverlay();
});
video.addEventListener("pause", () => {
  playPauseButton.textContent = "▶";
  playPauseButton.setAttribute("aria-label", "Reproducir");
  drawOverlay();
});
video.addEventListener("error", () => {
  setControlsEnabled(false);
  workspaceMessage.hidden = false;
  workspaceTitle.textContent = "No se pudo cargar el vídeo";
  workspaceDescription.textContent = "Elige otro MP4 en el selector.";
});

videoSelect.addEventListener("change", () => {
  const videoInfo = videoSelect.videos.find((item) => item.name === videoSelect.value);
  if (!videoInfo || videoInfo.name === selectedVideo?.name) return;
  if (reviewIsDirty && !window.confirm("¿Descartar los cambios sin guardar de este vídeo?")) {
    videoSelect.value = selectedVideo.name;
    return;
  }
  loadVideo(videoInfo);
});

saveButton.addEventListener("click", saveCurrentReview);

debugToggle.addEventListener("click", () => {
  debugOverlayEnabled = !debugOverlayEnabled;
  debugToggle.classList.toggle("is-active", debugOverlayEnabled);
  debugToggle.setAttribute("aria-pressed", String(debugOverlayEnabled));
  drawOverlay();
});

calibrationPanel.addEventListener("pointerdown", (event) => {
  event.stopPropagation();
});
calibrationPanel.addEventListener("pointerup", (event) => {
  event.stopPropagation();
});
calibrationPanel.addEventListener("pointermove", (event) => {
  event.stopPropagation();
});

calibrationPanelHeader.addEventListener("pointerdown", (event) => {
  event.stopPropagation();
  if (event.button !== 0 || event.target.closest("button")) return;
  calibrationPanelDrag = {
    pointerId: event.pointerId,
    startX: event.clientX,
    startY: event.clientY,
    position: { ...calibrationPanelPosition },
  };
  calibrationPanelHeader.setPointerCapture(event.pointerId);
  event.preventDefault();
});

calibrationPanelHeader.addEventListener("pointermove", (event) => {
  if (!calibrationPanelDrag || event.pointerId !== calibrationPanelDrag.pointerId) return;
  event.stopPropagation();
  positionCalibrationPanel({
    x: calibrationPanelDrag.position.x + event.clientX - calibrationPanelDrag.startX,
    y: calibrationPanelDrag.position.y + event.clientY - calibrationPanelDrag.startY,
  });
  event.preventDefault();
});

function finishCalibrationPanelDrag(event) {
  if (!calibrationPanelDrag || event.pointerId !== calibrationPanelDrag.pointerId) return;
  event.stopPropagation();
  calibrationPanelDrag = null;
  if (calibrationPanelHeader.hasPointerCapture(event.pointerId)) {
    calibrationPanelHeader.releasePointerCapture(event.pointerId);
  }
}

calibrationPanelHeader.addEventListener("pointerup", finishCalibrationPanelDrag);
calibrationPanelHeader.addEventListener("pointercancel", finishCalibrationPanelDrag);

calibrationPanelToggle.addEventListener("click", (event) => {
  event.stopPropagation();
  const maximumX = Math.max(0, workArea.clientWidth - calibrationPanel.offsetWidth);
  const moveToRight = calibrationPanelPosition.x <= maximumX / 2;
  positionCalibrationPanel({ x: moveToRight ? maximumX : 0, y: 60 });
});

workArea.addEventListener("pointermove", (event) => {
  pointerBrowserPosition = { clientX: event.clientX, clientY: event.clientY };
  pointerVideoCoordinates = browserToVideoCoordinates(pointerBrowserPosition);
  if (calibrationDragState && pointerVideoCoordinates) {
    const coordinates = clampToVideo(pointerVideoCoordinates);
    const { constraint, handle, origin, image } = calibrationDragState;
    if (constraint.type === "point") constraint.image = coordinates;
    if (constraint.type === "line") {
      if (handle === "start") constraint.image.start = coordinates;
      else if (handle === "end") constraint.image.end = coordinates;
      else {
        const deltaX = coordinates.x - origin.x;
        const deltaY = coordinates.y - origin.y;
        constraint.image.start = clampToVideo({ x: image.start.x + deltaX, y: image.start.y + deltaY });
        constraint.image.end = clampToVideo({ x: image.end.x + deltaX, y: image.end.y + deltaY });
      }
    }
    if (constraint.type === "arc") {
      const deltaX = coordinates.x - origin.x;
      const deltaY = coordinates.y - origin.y;
      const geometry = constraint.image.geometry;
      if (handle === "arc-radius-x" || handle === "arc-radius-y") {
        const relativeX = coordinates.x - geometry.center.x;
        const relativeY = coordinates.y - geometry.center.y;
        if (handle === "arc-radius-x") geometry.radiusX = Math.max(6, Math.abs(relativeX * Math.cos(geometry.rotation) + relativeY * Math.sin(geometry.rotation)));
        else geometry.radiusY = Math.max(6, Math.abs(-relativeX * Math.sin(geometry.rotation) + relativeY * Math.cos(geometry.rotation)));
      } else {
        constraint.image.geometry.center = clampToVideo({ x: image.geometry.center.x + deltaX, y: image.geometry.center.y + deltaY });
      }
    }
    syncLegacyPointCorrespondences(calibrationDragState.calibration);
    calibrationDragState.didMove = true;
    solveCalibration(calibrationDragState.calibration);
    drawOverlay();
    return;
  }
  if (calibrationMode && calibrationLineDraft && pointerVideoCoordinates) {
    calibrationLineDraft.end = clampToVideo(pointerVideoCoordinates);
    drawOverlay();
    return;
  }
  if (calibrationMode && calibrationArcDraft && pointerVideoCoordinates) {
    const coordinates = clampToVideo(pointerVideoCoordinates);
    const previous = calibrationArcDraft.samples[calibrationArcDraft.samples.length - 1];
    if (distanceBetween(previous, coordinates) >= 3 / (displayedVideoRect.width / video.videoWidth)) calibrationArcDraft.samples.push(coordinates);
    drawOverlay();
    return;
  }
  if (activeMode === "guided" && pointerVideoCoordinates) {
    if (lineDraft) {
      lineDraft.end = clampToVideo(pointerVideoCoordinates);
    } else if (assistedDraft) {
      const coordinates = clampToVideo(pointerVideoCoordinates);
      const previousSample = assistedDraft.samples[assistedDraft.samples.length - 1];
      const sampleSpacing = 3 / (displayedVideoRect.width / video.videoWidth);
      if (distanceBetween(coordinates, previousSample) >= sampleSpacing) assistedDraft.samples.push(coordinates);
    } else if (fieldDragState) {
      const annotation = findFieldAnnotation(fieldDragState.id);
      if (annotation) {
        const coordinates = clampToVideo(pointerVideoCoordinates);
        if (fieldDragState.handle === "point") annotation.coordinates = coordinates;
        if (fieldDragState.handle === "start") annotation.coordinates.start = coordinates;
        if (fieldDragState.handle === "end") annotation.coordinates.end = coordinates;
        if (fieldDragState.handle === "line") {
          const deltaX = coordinates.x - fieldDragState.origin.x;
          const deltaY = coordinates.y - fieldDragState.origin.y;
          annotation.coordinates.start = clampToVideo({
            x: fieldDragState.coordinates.start.x + deltaX,
            y: fieldDragState.coordinates.start.y + deltaY,
          });
          annotation.coordinates.end = clampToVideo({
            x: fieldDragState.coordinates.end.x + deltaX,
            y: fieldDragState.coordinates.end.y + deltaY,
          });
        }
        if (fieldDragState.handle === "ellipse-center") annotation.coordinates.center = coordinates;
        if (fieldDragState.handle === "ellipse-radius-x" || fieldDragState.handle === "ellipse-radius-y") {
          const { center, rotation } = annotation.coordinates;
          const relativeX = coordinates.x - center.x;
          const relativeY = coordinates.y - center.y;
          if (fieldDragState.handle === "ellipse-radius-x") {
            annotation.coordinates.radiusX = Math.max(6, Math.abs(relativeX * Math.cos(rotation) + relativeY * Math.sin(rotation)));
          } else {
            annotation.coordinates.radiusY = Math.max(6, Math.abs(-relativeX * Math.sin(rotation) + relativeY * Math.cos(rotation)));
          }
        }
        if (fieldDragState.handle === "ellipse") {
          const deltaX = coordinates.x - fieldDragState.origin.x;
          const deltaY = coordinates.y - fieldDragState.origin.y;
          annotation.coordinates.center = clampToVideo({
            x: fieldDragState.coordinates.center.x + deltaX,
            y: fieldDragState.coordinates.center.y + deltaY,
          });
        }
        fieldDragState.didMove = true;
      }
    }
  }
  drawOverlay();
});

workArea.addEventListener("pointerleave", () => {
  pointerBrowserPosition = null;
  pointerVideoCoordinates = null;
  drawOverlay();
});

workArea.addEventListener("pointerdown", (event) => {
  const pointerCoordinates = browserToVideoCoordinates(event);
  if (activeMode !== "guided" || !pointerCoordinates || !video.videoWidth) return;
  event.preventDefault();
  video.pause();
  const coordinates = clampToVideo(pointerCoordinates);

  if (calibrationMode) {
    const hit = calibrationHitTest(coordinates);
    if (hit) {
      beginCalibrationConstraintDrag(hit, coordinates, event.pointerId);
      drawOverlay();
    } else {
      if (calibrationType === "point") addCalibrationCorrespondence(coordinates);
      const selectedSemantic = selectedSemanticDefinition();
      if (calibrationType === "line" && selectedSemantic) {
        calibrationLineDraft = {
          semantic: selectedSemantic.id,
          start: coordinates,
          end: coordinates,
          pointerId: event.pointerId,
        };
        workArea.setPointerCapture(event.pointerId);
      }
      if (calibrationType === "arc" && selectedSemantic) {
        calibrationArcDraft = { semantic: selectedSemantic.id, samples: [coordinates], pointerId: event.pointerId };
        workArea.setPointerCapture(event.pointerId);
      }
    }
    return;
  }

  if (activeFieldTool === "select") {
    const calibrationHit = calibrationHitTest(coordinates);
    const fieldHit = hitTestFieldAnnotation(coordinates);
    if (calibrationHit && (isCalibrationHandle(calibrationHit) || !fieldHit)) {
      selectedFieldItem = null;
      beginCalibrationConstraintDrag(calibrationHit, coordinates, event.pointerId);
      drawOverlay();
      return;
    }
  }

  if (activeFieldTool === "point") {
    rememberFieldEdit();
    const annotation = createFieldAnnotation("point", coordinates);
    fieldAnnotations.push(annotation);
    selectedFieldItem = { id: annotation.id, handle: "point" };
    drawOverlay();
    return;
  }

  if (activeFieldTool === "line") {
    lineDraft = { start: coordinates, end: coordinates, pointerId: event.pointerId };
    workArea.setPointerCapture(event.pointerId);
    drawOverlay();
    return;
  }

  if (activeFieldTool === "smart-line" || activeFieldTool === "ellipse") {
    assistedDraft = { type: activeFieldTool, samples: [coordinates], pointerId: event.pointerId };
    workArea.setPointerCapture(event.pointerId);
    drawOverlay();
    return;
  }

  const hit = hitTestFieldAnnotation(coordinates);
  selectedCalibrationConstraint = null;
  selectedFieldItem = hit;
  if (hit) {
    fieldDragState = {
      ...hit,
      origin: coordinates,
      coordinates: cloneAnnotations().find((annotation) => annotation.id === hit.id).coordinates,
      before: cloneFieldState(),
      didMove: false,
    };
    workArea.setPointerCapture(event.pointerId);
  }
  drawOverlay();
});

workArea.addEventListener("pointerup", (event) => {
  if (calibrationDragState) {
    if (calibrationDragState.didMove) {
      fieldUndoStack.push(calibrationDragState.before);
      setReviewDirty(true);
    }
    calibrationDragState = null;
  }
  if (calibrationLineDraft?.pointerId === event.pointerId) {
    const draft = calibrationLineDraft;
    const coordinates = browserToVideoCoordinates(event);
    if (coordinates) draft.end = clampToVideo(coordinates);
    calibrationLineDraft = null;
    if (distanceBetween(draft.start, draft.end) > 2) {
      rememberFieldEdit();
      const calibration = ensureCurrentCalibration();
      const semantic = PITCH_MODEL.calibrationLinesById[draft.semantic];
      const constraints = calibrationConstraints(calibration);
      const constraint = { type: "line", semantic: semantic.id, world: semantic.world, image: { start: draft.start, end: draft.end }, source: "manual" };
      const existing = constraints.find((item) => item.type === "line" && item.semantic === semantic.id);
      if (existing) Object.assign(existing, constraint);
      else constraints.push(constraint);
      solveCalibration(calibration);
    }
  }
  if (calibrationArcDraft?.pointerId === event.pointerId) {
    const draft = calibrationArcDraft;
    const coordinates = browserToVideoCoordinates(event);
    if (coordinates && distanceBetween(draft.samples[draft.samples.length - 1], coordinates) > 0) draft.samples.push(clampToVideo(coordinates));
    calibrationArcDraft = null;
    if (draft.samples.length >= 6) {
      rememberFieldEdit();
      const calibration = ensureCurrentCalibration();
      const semantic = PITCH_MODEL.calibrationArcsById[draft.semantic];
      const constraints = calibrationConstraints(calibration);
      const constraint = {
        type: "arc", semantic: semantic.id, world: semantic.world,
        image: { samples: draft.samples, geometry: fitEllipse(draft.samples) }, source: "manual",
      };
      const existing = constraints.find((item) => item.type === "arc" && item.semantic === semantic.id);
      if (existing) Object.assign(existing, constraint);
      else constraints.push(constraint);
      solveCalibration(calibration);
    }
  }
  if (assistedDraft?.pointerId === event.pointerId) {
    const draft = assistedDraft;
    const pointerCoordinates = browserToVideoCoordinates(event);
    if (pointerCoordinates && distanceBetween(draft.samples[draft.samples.length - 1], pointerCoordinates) > 0) {
      draft.samples.push(clampToVideo(pointerCoordinates));
    }
    assistedDraft = null;
    if (draft.type === "smart-line" && draft.samples.length >= 2) {
      rememberFieldEdit();
      const annotation = createFieldAnnotation("line", fitSmartLine(draft.samples), {
        method: "pca_stroke_line",
        samples: draft.samples,
      });
      fieldAnnotations.push(annotation);
      selectedFieldItem = { id: annotation.id, handle: "line" };
    }
    if (draft.type === "ellipse" && draft.samples.length >= 6) {
      rememberFieldEdit();
      const annotation = createFieldAnnotation("ellipse", fitEllipse(draft.samples), {
        method: "radial_least_squares_ellipse",
        samples: draft.samples,
      });
      fieldAnnotations.push(annotation);
      selectedFieldItem = { id: annotation.id, handle: "ellipse" };
    }
  }
  if (lineDraft?.pointerId === event.pointerId) {
    const draft = lineDraft;
    const pointerCoordinates = browserToVideoCoordinates(event);
    if (pointerCoordinates) draft.end = clampToVideo(pointerCoordinates);
    lineDraft = null;
    if (distanceBetween(draft.start, draft.end) > 2) {
      rememberFieldEdit();
      const annotation = createFieldAnnotation("line", { start: draft.start, end: draft.end });
      fieldAnnotations.push(annotation);
      selectedFieldItem = { id: annotation.id, handle: "line" };
    }
  }
  if (fieldDragState) {
    if (fieldDragState.didMove) {
      fieldUndoStack.push(fieldDragState.before);
      setReviewDirty(true);
    }
    fieldDragState = null;
  }
  if (workArea.hasPointerCapture(event.pointerId)) workArea.releasePointerCapture(event.pointerId);
  drawOverlay();
});

fieldToolButtons.forEach((button) => {
  button.addEventListener("click", () => setActiveFieldTool(button.dataset.fieldTool));
});
scanVideoButton.addEventListener("click", startPitchSetup);
beginGuidedCalibrationButton.addEventListener("click", confirmPitchSetup);
openManualCalibrationButton.addEventListener("click", () => openPitchSetupMeasurement(pitchSetupCurrentFrame()));
guidedContextButton.addEventListener("click", playViewScanContext);
guidedSkipButton.addEventListener("click", skipViewScanRecommendation);
guidedConfirmContinueButton.addEventListener("click", completePitchSetupMeasurement);
exitCalibrationButton.addEventListener("click", exitCalibrationMode);
removeCalibrationPointButton.addEventListener("click", removeSelectedCalibrationCorrespondence);
clearCalibrationButton.addEventListener("click", clearCurrentCalibration);
calibrationConstraintType.addEventListener("change", () => {
  setCalibrationSelection(calibrationConstraintType.value);
  updateCalibrationStatus();
  drawOverlay();
});
calibrationLandmark.addEventListener("change", () => {
  setCalibrationSelection(calibrationType, calibrationLandmark.value);
  updateCalibrationStatus();
  drawOverlay();
});
deleteFieldItemButton?.addEventListener("click", deleteSelectedFieldItem);
undoFieldEditButton?.addEventListener("click", undoFieldEdit);
undoButton.addEventListener("click", undoFieldEdit);

new ResizeObserver(syncOverlay).observe(workArea);
window.addEventListener("resize", syncOverlay);

document.addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "z" && activeMode === "guided") {
    event.preventDefault();
    undoFieldEdit();
    return;
  }
  const targetName = event.target.tagName;
  if (targetName === "SELECT" || targetName === "BUTTON" || targetName === "INPUT") return;
  if (event.code === "Space") {
    event.preventDefault();
    if (!playPauseButton.disabled) playPauseButton.click();
  }
  if (video.paused && event.key === "ArrowLeft") {
    event.preventDefault();
    stepFrame(-1);
  }
  if (video.paused && event.key === "ArrowRight") {
    event.preventDefault();
    stepFrame(1);
  }
});

async function initialiseVideos() {
  try {
    const response = await fetch("/api/videos");
    const { videos } = await response.json();
    videoSelect.replaceChildren();
    videoSelect.videos = videos;

    if (!videos.length) {
      videoSelect.add(new Option("No se encontraron vídeos MP4", ""));
      workspaceTitle.textContent = "No hay vídeos disponibles";
      workspaceDescription.textContent = "Añade un archivo MP4 al directorio input.";
      return;
    }

    videos.forEach((videoInfo) => videoSelect.add(new Option(videoInfo.name, videoInfo.name)));
    videoSelect.disabled = false;
    loadVideo(videos[0]);
  } catch (error) {
    workspaceTitle.textContent = "Lista de vídeos no disponible";
    workspaceDescription.textContent = "No se pudo leer el directorio input.";
  }
}

initialiseVideos();
