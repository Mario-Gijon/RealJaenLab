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
const calibrationPanel = document.querySelector("#calibration-panel");
const calibrationLandmark = document.querySelector("#calibration-landmark");
const calibrationStatus = document.querySelector("#calibration-status");
const removeCalibrationPointButton = document.querySelector("#remove-calibration-point");
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

workArea.classList.add("field-editing");
PITCH_MODEL.landmarks.forEach((landmark) => calibrationLandmark.add(new Option(landmark.label, landmark.id)));

modeTabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    modeTabs.forEach((item) => item.classList.toggle("is-active", item === tab));
    activeMode = tab.dataset.mode;
    modeLabel.textContent = `${activeMode.toUpperCase()} MODE`;
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
  saveStatusLabel.textContent = isDirty ? "UNSAVED" : "SAVED";
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

function cloneAnnotations() {
  return JSON.parse(JSON.stringify(fieldAnnotations));
}

function cloneFieldState() {
  return JSON.parse(JSON.stringify({ annotations: fieldAnnotations, calibrations: fieldCalibrations }));
}

function rememberFieldEdit() {
  fieldUndoStack.push(cloneFieldState());
  setReviewDirty(true);
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
  if (video.readyState >= HTMLMediaElement.HAVE_METADATA) video.pause();
  drawOverlay();
}

function undoFieldEdit() {
  const previousState = fieldUndoStack.pop();
  if (!previousState) return;
  fieldAnnotations = previousState.annotations;
  fieldCalibrations = previousState.calibrations;
  selectedFieldItem = null;
  lineDraft = null;
  assistedDraft = null;
  calibrationDragState = null;
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

function updateCalibrationStatus(message = null) {
  const calibration = calibrationForCurrentFrame();
  if (message) {
    calibrationStatus.textContent = message;
    return;
  }
  const count = calibration?.correspondences.length || 0;
  calibrationStatus.textContent = count >= 4 && calibration?.homography
    ? `${count} landmarks — pitch projected`
    : `${count} landmarks — add at least 4 non-collinear points`;
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
  const correspondence = calibration.correspondences.find((item) => distanceBetween(item.image, coordinates) <= hitRadius);
  return correspondence ? { calibration, correspondence } : null;
}

async function solveCalibration(calibration) {
  if (calibration.correspondences.length < 4) {
    calibration.homography = null;
    calibration.reprojectionError = null;
    updateCalibrationStatus();
    drawOverlay();
    return;
  }
  const sequence = ++calibrationSolveSequence;
  updateCalibrationStatus("Calculating pitch projection…");
  try {
    const response = await fetch("/api/homography", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ correspondences: calibration.correspondences }),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || "Unable to calculate pitch projection");
    if (sequence !== calibrationSolveSequence || !fieldCalibrations.includes(calibration)) return;
    calibration.homography = result.homography;
    calibration.reprojectionError = result.reprojection_error;
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
  const landmark = PITCH_MODEL.landmarksById[calibrationLandmark.value];
  if (!landmark) return;
  rememberFieldEdit();
  const calibration = ensureCurrentCalibration();
  const existing = calibration.correspondences.find((item) => item.landmark === landmark.id);
  const correspondence = {
    landmark: landmark.id,
    world: { ...landmark.world },
    image: clampToVideo(image),
  };
  if (existing) Object.assign(existing, correspondence);
  else calibration.correspondences.push(correspondence);
  calibration.homography = null;
  solveCalibration(calibration);
}

function removeSelectedCalibrationCorrespondence() {
  const calibration = calibrationForCurrentFrame();
  if (!calibration) return;
  const landmarkId = calibrationLandmark.value;
  if (!calibration.correspondences.some((item) => item.landmark === landmarkId)) return;
  rememberFieldEdit();
  calibration.correspondences = calibration.correspondences.filter((item) => item.landmark !== landmarkId);
  calibration.homography = null;
  solveCalibration(calibration);
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

function projectWorldPoint(point, homography) {
  const [first, second, third] = homography;
  const denominator = third[0] * point.x + third[1] * point.y + third[2];
  if (Math.abs(denominator) < 1e-8) return null;
  return {
    x: (first[0] * point.x + first[1] * point.y + first[2]) / denominator,
    y: (second[0] * point.x + second[1] * point.y + second[2]) / denominator,
  };
}

function drawProjectedPath(points, homography, displayScale) {
  let started = false;
  overlayContext.beginPath();
  points.forEach((point) => {
    const projected = projectWorldPoint(point, homography);
    if (!projected) return;
    if (!started) {
      overlayContext.moveTo(projected.x, projected.y);
      started = true;
    } else {
      overlayContext.lineTo(projected.x, projected.y);
    }
  });
  if (started) overlayContext.stroke();
}

function drawCalibrationOverlay() {
  const calibration = calibrationForCurrentFrame();
  if (!calibration || !displayedVideoRect || !video.videoWidth) return;
  const displayScale = displayedVideoRect.width / video.videoWidth;
  overlayContext.save();

  if (Array.isArray(calibration.homography)) {
    overlayContext.strokeStyle = "#5aa9ff";
    overlayContext.lineWidth = 2 / displayScale;
    PITCH_MODEL.segments.forEach((segment) => drawProjectedPath(segment, calibration.homography, displayScale));
    drawProjectedPath(PITCH_MODEL.sampleCircle({ x: 0, y: 0 }, PITCH_MODEL.centerCircleRadius), calibration.homography, displayScale);
    [
      { x: 0, y: 0 },
      { x: -41.5, y: 0 },
      { x: 41.5, y: 0 },
    ].forEach((worldPoint) => {
      const point = projectWorldPoint(worldPoint, calibration.homography);
      if (!point) return;
      overlayContext.beginPath();
      overlayContext.arc(point.x, point.y, 3.5 / displayScale, 0, Math.PI * 2);
      overlayContext.fillStyle = "#5aa9ff";
      overlayContext.fill();
    });
  }

  calibration.correspondences.forEach((correspondence) => {
    const selected = calibrationMode && correspondence.landmark === calibrationLandmark.value;
    const radius = (selected ? 9 : 7) / displayScale;
    overlayContext.beginPath();
    overlayContext.arc(correspondence.image.x, correspondence.image.y, radius, 0, Math.PI * 2);
    overlayContext.fillStyle = selected ? "#f0f2f4" : "#5aa9ff";
    overlayContext.fill();
    overlayContext.strokeStyle = "#102d4d";
    overlayContext.lineWidth = 2 / displayScale;
    overlayContext.stroke();
  });
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
    frameReadout.textContent = `Frame ${currentFrame} / ${totalFrames}`;
  } else {
    frameReadout.textContent = "Frame —";
  }
}

function resetPlaybackState() {
  video.pause();
  video.currentTime = 0;
  playPauseButton.textContent = "▶";
  playPauseButton.setAttribute("aria-label", "Play");
  timeline.max = 0;
  timeline.value = 0;
  currentTimeLabel.textContent = "00:00:00";
  totalTimeLabel.textContent = "00:00:00";
  frameReadout.textContent = "Frame —";
  pointerVideoCoordinates = null;
  pointerBrowserPosition = null;
  displayedVideoRect = null;
  overlayContext.clearRect(0, 0, overlay.width, overlay.height);
  overlay.classList.remove("is-loaded");
}

async function loadVideo(videoInfo) {
  const loadSequence = ++videoLoadSequence;
  selectedVideo = videoInfo;
  fieldAnnotations = [];
  fieldCalibrations = [];
  fieldUndoStack = [];
  selectedFieldItem = null;
  lineDraft = null;
  assistedDraft = null;
  calibrationMode = false;
  calibrationPanel.hidden = true;
  calibrationDragState = null;
  setReviewDirty(false);
  resetPlaybackState();
  setControlsEnabled(false);
  video.classList.remove("is-loaded");
  workspaceMessage.hidden = false;
  workspaceTitle.textContent = "Loading video";
  workspaceDescription.textContent = videoInfo.name;
  video.src = `/videos/${encodeURIComponent(videoInfo.name)}`;
  video.load();

  try {
    const response = await fetch(`/api/reviews/${encodeURIComponent(videoInfo.name)}`);
    if (!response.ok) throw new Error("Unable to load review");
    const { review } = await response.json();
    if (loadSequence !== videoLoadSequence) return;
    fieldAnnotations = Array.isArray(review?.field?.annotations) ? review.field.annotations : [];
    fieldCalibrations = Array.isArray(review?.field?.calibrations) ? review.field.calibrations : [];
    annotationSequence = fieldAnnotations.length;
    setReviewDirty(false);
    drawOverlay();
  } catch (error) {
    if (loadSequence !== videoLoadSequence) return;
    setReviewDirty(true);
    workspaceDescription.textContent = "Review could not be loaded. You can still create and save one.";
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
        calibration.video === selectedVideo.name && Array.isArray(calibration.homography)
      )),
    },
    people: {},
  };

  try {
    const response = await fetch(`/api/reviews/${encodeURIComponent(selectedVideo.name)}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(review),
    });
    if (!response.ok) throw new Error("Unable to save review");
    setReviewDirty(false);
  } catch (error) {
    setReviewDirty(true);
    saveStatusLabel.textContent = "SAVE FAILED";
  } finally {
    saveButton.disabled = false;
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
  drawOverlay();
});
video.addEventListener("play", () => {
  playPauseButton.textContent = "❚❚";
  playPauseButton.setAttribute("aria-label", "Pause");
  drawOverlay();
});
video.addEventListener("pause", () => {
  playPauseButton.textContent = "▶";
  playPauseButton.setAttribute("aria-label", "Play");
  drawOverlay();
});
video.addEventListener("error", () => {
  setControlsEnabled(false);
  workspaceMessage.hidden = false;
  workspaceTitle.textContent = "Unable to load video";
  workspaceDescription.textContent = "Choose another MP4 from the selector.";
});

videoSelect.addEventListener("change", () => {
  const videoInfo = videoSelect.videos.find((item) => item.name === videoSelect.value);
  if (!videoInfo || videoInfo.name === selectedVideo?.name) return;
  if (reviewIsDirty && !window.confirm("Discard unsaved changes for this video?")) {
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
  if (calibrationMode && calibrationDragState && pointerVideoCoordinates) {
    calibrationDragState.correspondence.image = clampToVideo(pointerVideoCoordinates);
    calibrationDragState.didMove = true;
    solveCalibration(calibrationDragState.calibration);
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
      calibrationLandmark.value = hit.correspondence.landmark;
      calibrationDragState = {
        ...hit,
        before: cloneFieldState(),
        didMove: false,
      };
      workArea.setPointerCapture(event.pointerId);
      updateCalibrationStatus();
      drawOverlay();
    } else {
      addCalibrationCorrespondence(coordinates);
    }
    return;
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
exitCalibrationButton.addEventListener("click", exitCalibrationMode);
removeCalibrationPointButton.addEventListener("click", removeSelectedCalibrationCorrespondence);
calibrationLandmark.addEventListener("change", () => {
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
      videoSelect.add(new Option("No MP4 videos found", ""));
      workspaceTitle.textContent = "No videos available";
      workspaceDescription.textContent = "Add an MP4 file to the input directory.";
      return;
    }

    videos.forEach((videoInfo) => videoSelect.add(new Option(videoInfo.name, videoInfo.name)));
    videoSelect.disabled = false;
    loadVideo(videos[0]);
  } catch (error) {
    workspaceTitle.textContent = "Video list unavailable";
    workspaceDescription.textContent = "Unable to read the input directory.";
  }
}

initialiseVideos();
