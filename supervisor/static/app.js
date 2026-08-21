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

let selectedVideo = null;
let debugOverlayEnabled = false;
let pointerVideoCoordinates = null;
let pointerBrowserPosition = null;
let displayedVideoRect = null;
let activeMode = "field";
let activeFieldTool = "select";
let fieldAnnotations = [];
let fieldUndoStack = [];
let selectedFieldItem = null;
let fieldDragState = null;
let lineDraft = null;
let annotationSequence = 0;

workArea.classList.add("field-editing");

modeTabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    modeTabs.forEach((item) => item.classList.toggle("is-active", item === tab));
    activeMode = tab.dataset.mode;
    modeLabel.textContent = `${activeMode.toUpperCase()} MODE`;
    fieldToolbar.hidden = activeMode !== "field";
    workArea.classList.toggle("field-editing", activeMode === "field");
    selectedFieldItem = null;
    lineDraft = null;
    drawOverlay();
  });
});

saveButton.addEventListener("click", () => {
  saveStatus.classList.add("is-saved");
  saveStatusLabel.textContent = "SAVED";
});

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

function cloneAnnotations() {
  return JSON.parse(JSON.stringify(fieldAnnotations));
}

function rememberFieldEdit() {
  fieldUndoStack.push(cloneAnnotations());
}

function clampToVideo({ x, y }) {
  return {
    x: Math.min(Math.max(x, 0), video.videoWidth),
    y: Math.min(Math.max(y, 0), video.videoHeight),
  };
}

function createFieldAnnotation(type, coordinates) {
  annotationSequence += 1;
  return {
    id: `field-${Date.now()}-${annotationSequence}`,
    video: selectedVideo.name,
    frame: currentFrameNumber(),
    type,
    coordinates,
    source: "manual",
  };
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
  }
  return null;
}

function setActiveFieldTool(tool) {
  activeFieldTool = tool;
  fieldToolButtons.forEach((button) => button.classList.toggle("is-active", button.dataset.fieldTool === tool));
  lineDraft = null;
  if (video.readyState >= HTMLMediaElement.HAVE_METADATA) video.pause();
  drawOverlay();
}

function undoFieldEdit() {
  const previousAnnotations = fieldUndoStack.pop();
  if (!previousAnnotations) return;
  fieldAnnotations = previousAnnotations;
  selectedFieldItem = null;
  lineDraft = null;
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

function loadVideo(videoInfo) {
  selectedVideo = videoInfo;
  resetPlaybackState();
  setControlsEnabled(false);
  video.classList.remove("is-loaded");
  workspaceMessage.hidden = false;
  workspaceTitle.textContent = "Loading video";
  workspaceDescription.textContent = videoInfo.name;
  video.src = `/videos/${encodeURIComponent(videoInfo.name)}`;
  video.load();
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
  if (videoInfo) loadVideo(videoInfo);
});

debugToggle.addEventListener("click", () => {
  debugOverlayEnabled = !debugOverlayEnabled;
  debugToggle.classList.toggle("is-active", debugOverlayEnabled);
  debugToggle.setAttribute("aria-pressed", String(debugOverlayEnabled));
  drawOverlay();
});

workArea.addEventListener("pointermove", (event) => {
  pointerBrowserPosition = { clientX: event.clientX, clientY: event.clientY };
  pointerVideoCoordinates = browserToVideoCoordinates(pointerBrowserPosition);
  if (activeMode === "field" && pointerVideoCoordinates) {
    if (lineDraft) {
      lineDraft.end = clampToVideo(pointerVideoCoordinates);
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

  const hit = hitTestFieldAnnotation(coordinates);
  selectedFieldItem = hit;
  if (hit) {
    fieldDragState = {
      ...hit,
      origin: coordinates,
      coordinates: cloneAnnotations().find((annotation) => annotation.id === hit.id).coordinates,
      before: cloneAnnotations(),
      didMove: false,
    };
    workArea.setPointerCapture(event.pointerId);
  }
  drawOverlay();
});

workArea.addEventListener("pointerup", (event) => {
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
    if (fieldDragState.didMove) fieldUndoStack.push(fieldDragState.before);
    fieldDragState = null;
  }
  if (workArea.hasPointerCapture(event.pointerId)) workArea.releasePointerCapture(event.pointerId);
  drawOverlay();
});

fieldToolButtons.forEach((button) => {
  button.addEventListener("click", () => setActiveFieldTool(button.dataset.fieldTool));
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
