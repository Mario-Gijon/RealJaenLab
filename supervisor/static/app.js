const modeTabs = document.querySelectorAll(".mode-tab");
const modeLabel = document.querySelector("#mode-label");
const saveButton = document.querySelector("#save-button");
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

let selectedVideo = null;
let debugOverlayEnabled = false;
let pointerVideoCoordinates = null;
let pointerBrowserPosition = null;
let displayedVideoRect = null;

modeTabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    modeTabs.forEach((item) => item.classList.toggle("is-active", item === tab));
    modeLabel.textContent = `${tab.dataset.mode.toUpperCase()} MODE`;
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

function drawDebugOverlay() {
  overlayContext.clearRect(0, 0, overlay.width, overlay.height);
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
  drawDebugOverlay();
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
  drawDebugOverlay();
});
video.addEventListener("seeked", () => {
  updatePlaybackDisplay();
  drawDebugOverlay();
});
video.addEventListener("play", () => {
  playPauseButton.textContent = "❚❚";
  playPauseButton.setAttribute("aria-label", "Pause");
  drawDebugOverlay();
});
video.addEventListener("pause", () => {
  playPauseButton.textContent = "▶";
  playPauseButton.setAttribute("aria-label", "Play");
  drawDebugOverlay();
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
  drawDebugOverlay();
});

workArea.addEventListener("pointermove", (event) => {
  pointerBrowserPosition = { clientX: event.clientX, clientY: event.clientY };
  pointerVideoCoordinates = browserToVideoCoordinates(pointerBrowserPosition);
  drawDebugOverlay();
});

workArea.addEventListener("pointerleave", () => {
  pointerBrowserPosition = null;
  pointerVideoCoordinates = null;
  drawDebugOverlay();
});

new ResizeObserver(syncOverlay).observe(workArea);
window.addEventListener("resize", syncOverlay);

document.addEventListener("keydown", (event) => {
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
