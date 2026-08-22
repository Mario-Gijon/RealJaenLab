const modeTabs = document.querySelectorAll(".mode-tab");
const modeLabel = document.querySelector("#mode-label");
const saveButton = document.querySelector("#save-button");
const undoButton = document.querySelector("#undo-button");
const saveStatus = document.querySelector("#save-status");
const saveStatusLabel = document.querySelector("#save-status-label");
const video = document.querySelector("#video");
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
const fieldToolbar = document.querySelector("#field-toolbar");
const fieldToolButtons = document.querySelectorAll("[data-field-tool]");
const deleteFieldItemButton = document.querySelector("#delete-field-item");
const undoFieldEditButton = document.querySelector("#undo-field-edit");
const calibratePitchButton = document.querySelector("#calibrate-pitch");
const propagateFieldButton = document.querySelector("#propagate-field");
const refineSegmentButton = document.querySelector("#refine-segment");
const propagationStatus = document.querySelector("#field-propagation-status");
const calibrationPanel = document.querySelector("#calibration-panel");
const calibrationLandmark = document.querySelector("#calibration-landmark");
const calibrationPitchMarkers = document.querySelector("#calibration-pitch-markers");
const calibrationPitchLines = document.querySelector("#calibration-pitch-lines");
const calibrationPitchArcs = document.querySelector("#calibration-pitch-arcs");
const calibrationStatus = document.querySelector("#calibration-status");
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
let activeMode = "field";
let activeFieldTool = "select";
let fieldAnnotations = [];
let fieldCalibrations = [];
let fieldPropagations = [];
let fieldRefinedSegments = [];
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
let calibrationType = "point";
let calibrationSemantic = PITCH_MODEL.landmarks[0]?.id || null;
const selectedSemanticByType = { point: calibrationSemantic, line: null, arc: null };
let calibrationLineDraft = null;
let calibrationArcDraft = null;
let selectedCalibrationConstraint = null;
let propagationInProgress = false;
let propagationRequestSequence = 0;
let refinementInProgress = false;
let refinementRequestSequence = 0;
let manualCalibrationRevision = 0;

workArea.classList.add("field-editing");

modeTabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    modeTabs.forEach((item) => item.classList.toggle("is-active", item === tab));
    activeMode = tab.dataset.mode;
    modeLabel.textContent = activeMode === "field" ? "MODO CAMPO" : "MODO PERSONAS";
    fieldToolbar.hidden = activeMode !== "field";
    workArea.classList.toggle("field-editing", activeMode === "field");
    selectedFieldItem = null;
    lineDraft = null;
    assistedDraft = null;
    calibrationMode = false;
    calibrationPanel.hidden = true;
    drawOverlay();
  });
});

function setReviewDirty(isDirty) {
  reviewIsDirty = isDirty;
  saveStatus.classList.toggle("is-saved", !isDirty);
  saveStatusLabel.textContent = isDirty ? "SIN GUARDAR" : "GUARDADO";
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

function hasSolvedManualCalibration(calibration = calibrationForCurrentFrame()) {
  return calibration?.source === "manual"
    && Array.isArray(calibration.homography)
    && calibration.homography.length === 3;
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

function setPropagationStatus(message = "") {
  propagationStatus.textContent = message;
  propagationStatus.hidden = !message;
}

function updatePropagationControl() {
  const enabled = !propagationInProgress && !refinementInProgress && hasSolvedManualCalibration();
  propagateFieldButton.disabled = !enabled;
  propagateFieldButton.textContent = propagationInProgress ? "Propagando campo…" : "Propagar 5 s →";
}

function isFieldEditInProgress() {
  return Boolean(
    fieldDragState || calibrationDragState || lineDraft || assistedDraft || calibrationLineDraft || calibrationArcDraft
  );
}

function previousManualCalibration(calibration = calibrationForCurrentFrame()) {
  if (!calibration || !selectedVideo) return null;
  return fieldCalibrations
    .filter((item) => (
      item.video === selectedVideo.name
      && item.frame < calibration.frame
      && hasSolvedManualCalibration(item)
    ))
    .sort((first, second) => second.frame - first.frame)[0] || null;
}

function updateRefinementControl() {
  const enabled = !propagationInProgress
    && !refinementInProgress
    && !isFieldEditInProgress()
    && hasSolvedManualCalibration()
    && Boolean(previousManualCalibration());
  refineSegmentButton.disabled = !enabled;
  refineSegmentButton.textContent = refinementInProgress ? "Refinando tramo FIELD…" : "Refinar tramo anterior";
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
  calibrationMode = false;
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
  if (activeMode !== "field" || !selectedFieldItem) return;
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

function renderCalibrationPitch() {
  const svgNamespace = "http://www.w3.org/2000/svg";
  const calibration = calibrationForCurrentFrame();
  const constraints = calibrationConstraints(calibration);
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

function updateCalibrationStatus(message = null) {
  syncSemanticSelector();
  const calibration = calibrationForCurrentFrame();
  renderCalibrationPitch();
  updateCurrentReferenceLabel();
  calibrationHint.textContent = calibrationType === "point"
    ? "Elige un punto en este diagrama y después haz clic en el mismo punto del vídeo."
    : calibrationType === "line"
      ? "Elige una línea del diagrama y arrastra un tramo visible de esa línea en el vídeo."
      : "Elige el arco del diagrama y traza su parte visible en el vídeo.";
  if (message) {
    calibrationStatus.textContent = `${constraintCountsLabel(calibrationConstraints(calibration))} · ${message}`;
    return;
  }
  const constraints = calibrationConstraints(calibration);
  const lineWarning = calibrationLineDiagnostics(calibration).find((diagnostic) => (
    diagnostic.quality === "bad" || (!diagnostic.quality && diagnostic.errorPx > LINE_WARNING_ERROR_PX)
  ));
  const hasMixedConstraints = constraints.some((item) => item.type !== "point");
  calibrationStatus.textContent = calibration?.homography
    ? `${constraintCountsLabel(constraints)} · campo proyectado${lineWarning ? ` · ${lineWarning.label} no está siendo respetada (error: ${Math.round(lineWarning.errorPx)} px)` : ""}`
    : hasMixedConstraints
      ? `${constraintCountsLabel(constraints)} · solver mixto pendiente`
      : `${constraintCountsLabel(constraints)} / mínimo 4`;
}

function enterCalibrationMode() {
  if (!selectedVideo) return;
  video.pause();
  calibrationMode = true;
  activeFieldTool = "select";
  fieldToolButtons.forEach((button) => button.classList.remove("is-active"));
  calibrationPanel.hidden = false;
  updateCalibrationStatus();
  drawOverlay();
}

function exitCalibrationMode() {
  calibrationMode = false;
  calibrationDragState = null;
  calibrationLineDraft = null;
  calibrationArcDraft = null;
  calibrationPanel.hidden = true;
  setActiveFieldTool("select");
  drawOverlay();
}

function ensureCurrentCalibration() {
  let calibration = calibrationForCurrentFrame();
  if (calibration) return calibration;
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
  manualCalibrationRevision += 1;
  invalidatePropagationsFromAnchor(calibration);
  invalidateRefinedSegmentsForCalibration(calibration);
  const constraints = calibrationConstraints(calibration);
  const pointCount = constraints.filter((item) => item.type === "point").length;
  if (pointCount < 4 && !constraints.some((item) => item.type !== "point")) {
    calibration.homography = null;
    calibration.reprojectionError = null;
    updateCalibrationStatus();
    drawOverlay();
    return;
  }
  const sequence = ++calibrationSolveSequence;
  updateCalibrationStatus("Calculando la proyección del campo…");
  try {
    const response = await fetch("/api/field-calibration/solve", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ constraints, image_width: video.videoWidth, image_height: video.videoHeight }),
    });
    const result = await response.json();
    const solved = result.status === "solved" || result.status === "solved_with_warnings";
    if (!response.ok || !solved) throw new Error(result.detail || "No se pudo calcular la proyección del campo");
    if (sequence !== calibrationSolveSequence || !fieldCalibrations.includes(calibration)) return;
    calibration.homography = result.homography;
    calibration.reprojectionError = result.reprojection_error;
    calibration.camera = result.camera || null;
    calibration.diagnostics = result.diagnostics || null;
    updateCalibrationStatus();
    drawOverlay();
  } catch (error) {
    if (sequence !== calibrationSolveSequence || !fieldCalibrations.includes(calibration)) return;
    calibration.homography = null;
    calibration.reprojectionError = null;
    updateCalibrationStatus(error.message);
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
  calibration.homography = null;
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
  calibration.homography = null;
  solveCalibration(calibration);
}

function clearCurrentCalibration() {
  const calibration = calibrationForCurrentFrame();
  if (!calibration) return;
  rememberFieldEdit();
  invalidatePropagationsFromAnchor(calibration);
  invalidateRefinedSegmentsForCalibration(calibration);
  manualCalibrationRevision += 1;
  fieldCalibrations = fieldCalibrations.filter((item) => item.id !== calibration.id);
  calibrationSolveSequence += 1;
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

function drawProjectedPitch(homography, displayScale) {
  overlayContext.strokeStyle = "#5aa9ff";
  overlayContext.lineWidth = 2 / displayScale;
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
  [
    { x: 0, y: 0 },
    { x: -41.5, y: 0 },
    { x: 41.5, y: 0 },
  ].forEach((worldPoint) => {
    const point = projectWorldPoint(worldPoint, homography);
    if (!point) return;
    overlayContext.beginPath();
    overlayContext.arc(point.x, point.y, 3.5 / displayScale, 0, Math.PI * 2);
    overlayContext.fillStyle = "#5aa9ff";
    overlayContext.fill();
  });
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
  drawFieldAnnotations();
  drawCalibrationOverlay();
  drawDebugOverlay();
  updatePropagationControl();
  updateRefinementControl();
  updateCurrentReferenceLabel();
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
  propagationRequestSequence += 1;
  refinementRequestSequence += 1;
  propagationInProgress = false;
  refinementInProgress = false;
  selectedVideo = videoInfo;
  fieldAnnotations = [];
  fieldCalibrations = [];
  fieldPropagations = [];
  fieldRefinedSegments = [];
  fieldUndoStack = [];
  selectedFieldItem = null;
  selectedCalibrationConstraint = null;
  lineDraft = null;
  assistedDraft = null;
  calibrationMode = false;
  calibrationPanel.hidden = true;
  calibrationDragState = null;
  calibrationLineDraft = null;
  calibrationArcDraft = null;
  setPropagationStatus();
  updatePropagationControl();
  updateRefinementControl();
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
    fieldCalibrations.forEach((calibration) => calibrationConstraints(calibration));
    annotationSequence = fieldAnnotations.length;
    setReviewDirty(false);
    drawOverlay();
  } catch (error) {
    if (loadSequence !== videoLoadSequence) return;
    setReviewDirty(true);
    workspaceDescription.textContent = "No se pudo cargar la revisión. Puedes crear una y guardarla.";
  }
}

async function saveCurrentReview() {
  if (!selectedVideo) return;
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
  } catch (error) {
    setReviewDirty(true);
    saveStatusLabel.textContent = "ERROR AL GUARDAR";
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
  propagationInProgress = true;
  setPropagationStatus("Propagando campo…");
  updatePropagationControl();

  try {
    const response = await fetch("/api/field-propagation", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        filename,
        anchor_frame: anchorFrame,
        homography,
        max_frames: 150,
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

    if (result.status === "partial" && frames.length) {
      const lastFrame = frames[frames.length - 1].frame;
      setPropagationStatus(`Propagado hasta el fotograma ${lastFrame}. Se necesita revisión: ${result.detail || "confianza insuficiente."}`);
    } else if (frames.length) {
      setPropagationStatus(`Campo propagado durante ${frames.length} fotogramas.`);
    } else {
      setPropagationStatus(`No se pudo propagar el campo. Se necesita revisión: ${result.detail || "sin fotogramas fiables."}`);
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
});

video.addEventListener("timeupdate", () => {
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
  if (activeMode === "field" && pointerVideoCoordinates) {
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
  if (activeMode !== "field" || !pointerCoordinates || !video.videoWidth) return;
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
calibratePitchButton.addEventListener("click", enterCalibrationMode);
propagateFieldButton.addEventListener("click", propagateCurrentCalibration);
refineSegmentButton.addEventListener("click", refinePreviousSegment);
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
deleteFieldItemButton.addEventListener("click", deleteSelectedFieldItem);
undoFieldEditButton.addEventListener("click", undoFieldEdit);
undoButton.addEventListener("click", undoFieldEdit);

new ResizeObserver(syncOverlay).observe(workArea);
window.addEventListener("resize", syncOverlay);

document.addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "z" && activeMode === "field") {
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
