const video = document.querySelector("#video");
const libraryScreen = document.querySelector("#library-screen");
const backToLibraryButton = document.querySelector("#back-to-library");
const videoName = document.querySelector("#video-name");
const workspaceKicker = document.querySelector("#workspace-kicker");
const workspaceTabs = document.querySelectorAll(".workspace-tab");
const videoWorkspace = document.querySelector("#video-workspace");
const previewWorkspace = document.querySelector("#preview-workspace");
const calibrationWorkspace = document.querySelector("#calibration-workspace");
const videoControls = document.querySelector("#video-controls");
const videoEmpty = document.querySelector("#video-empty");
const workspaceTitle = document.querySelector("#workspace-title");
const workspaceDescription = document.querySelector("#workspace-description");
const videoInfo = document.querySelector("#video-info");
const videoInfoName = document.querySelector("#video-info-name");
const videoInfoMeta = document.querySelector("#video-info-meta");
const playPauseButton = document.querySelector("#play-pause");
const muteToggleButton = document.querySelector("#mute-toggle");
const previousFrameButtons = document.querySelectorAll("#previous-frame, #previous-frame-label");
const nextFrameButtons = document.querySelectorAll("#next-frame, #next-frame-label");
const timeline = document.querySelector("#timeline");
const currentTimeLabel = document.querySelector("#current-time");
const totalTimeLabel = document.querySelector("#total-time");
const frameReadout = document.querySelector("#frame-readout");

let selectedVideo = null;
let activeWorkspace = "video";

function setWorkspace(workspace) {
  activeWorkspace = workspace;
  const isVideo = workspace === "video";
  videoWorkspace.hidden = !isVideo;
  previewWorkspace.hidden = workspace !== "preview";
  calibrationWorkspace.hidden = workspace !== "calibration";
  videoControls.hidden = !isVideo;
  workspaceKicker.textContent = workspace === "calibration" ? "CALIBRACIÓN" : workspace.toUpperCase();
  workspaceTabs.forEach((tab) => {
    const selected = tab.dataset.workspace === workspace;
    tab.classList.toggle("is-active", selected);
    tab.setAttribute("aria-current", selected ? "page" : "false");
  });
}

function formatTime(seconds) {
  const value = Math.max(0, Math.floor(seconds || 0));
  return [Math.floor(value / 3600), Math.floor((value % 3600) / 60), value % 60]
    .map((part) => String(part).padStart(2, "0"))
    .join(":");
}

function setControlsEnabled(enabled) {
  playPauseButton.disabled = !enabled;
  muteToggleButton.disabled = !enabled;
  timeline.disabled = !enabled;
  previousFrameButtons.forEach((button) => { button.disabled = !enabled; });
  nextFrameButtons.forEach((button) => { button.disabled = !enabled; });
}

function updateMuteControl() {
  const muted = video.muted;
  muteToggleButton.textContent = muted ? "🔇" : "🔊";
  muteToggleButton.setAttribute("aria-pressed", String(muted));
  muteToggleButton.setAttribute("aria-label", muted ? "Activar sonido" : "Silenciar");
}

function updatePlaybackDisplay() {
  const duration = Number.isFinite(video.duration) ? video.duration : (selectedVideo?.duration || 0);
  const currentTime = video.currentTime || 0;
  timeline.max = duration;
  timeline.value = Math.min(currentTime, duration);
  currentTimeLabel.textContent = formatTime(currentTime);
  totalTimeLabel.textContent = formatTime(duration);
  if (selectedVideo?.fps) {
    const frame = Math.min(Math.floor(currentTime * selectedVideo.fps), Math.max(0, (selectedVideo.frame_count || Infinity) - 1));
    frameReadout.textContent = `Fotograma ${frame} / ${selectedVideo.frame_count ?? "—"}`;
  } else {
    frameReadout.textContent = "Fotograma —";
  }
}

function stepFrame(direction) {
  if (!selectedVideo?.fps) return;
  video.pause();
  const duration = Number.isFinite(video.duration) ? video.duration : Infinity;
  video.currentTime = Math.max(0, Math.min(duration, video.currentTime + direction / selectedVideo.fps));
}

function showLibrary() {
  video.pause();
  document.body.classList.add("is-library");
  libraryScreen.hidden = false;
  window.SupervisorLibrary?.show();
}

function openVideo(videoInfo) {
  selectedVideo = videoInfo;
  document.body.classList.remove("is-library");
  libraryScreen.hidden = true;
  setControlsEnabled(false);
  video.muted = true;
  updateMuteControl();
  videoName.textContent = videoInfo.displayName || videoInfo.name;
  videoInfoName.textContent = videoInfo.displayName || videoInfo.name;
  videoInfoMeta.textContent = "Cargando metadatos…";
  videoInfo.hidden = false;
  videoEmpty.hidden = false;
  workspaceTitle.textContent = "Cargando vídeo";
  workspaceDescription.textContent = videoInfo.displayName || videoInfo.name;
  video.src = videoInfo.playbackUrl;
  video.load();
}

video.addEventListener("loadedmetadata", () => {
  videoEmpty.hidden = true;
  setControlsEnabled(true);
  const resolution = video.videoWidth && video.videoHeight ? `${video.videoWidth} × ${video.videoHeight}` : "Resolución desconocida";
  const fps = selectedVideo?.fps ? `${Number(selectedVideo.fps).toFixed(2)} fps` : "fps desconocidos";
  videoInfoMeta.textContent = `${resolution} · ${fps}`;
  updatePlaybackDisplay();
});
video.addEventListener("timeupdate", updatePlaybackDisplay);
video.addEventListener("seeked", updatePlaybackDisplay);
video.addEventListener("play", () => { playPauseButton.textContent = "❚❚"; playPauseButton.setAttribute("aria-label", "Pausar"); });
video.addEventListener("pause", () => { playPauseButton.textContent = "▶"; playPauseButton.setAttribute("aria-label", "Reproducir"); });
video.addEventListener("volumechange", updateMuteControl);
video.addEventListener("error", () => {
  setControlsEnabled(false);
  videoEmpty.hidden = false;
  workspaceTitle.textContent = "No se pudo cargar el vídeo";
  workspaceDescription.textContent = "Vuelve a la biblioteca para elegir otro archivo.";
});

playPauseButton.addEventListener("click", () => { if (video.paused) void video.play(); else video.pause(); });
muteToggleButton.addEventListener("click", () => { video.muted = !video.muted; });
previousFrameButtons.forEach((button) => button.addEventListener("click", () => stepFrame(-1)));
nextFrameButtons.forEach((button) => button.addEventListener("click", () => stepFrame(1)));
timeline.addEventListener("input", () => { video.currentTime = Number(timeline.value); });
backToLibraryButton.addEventListener("click", showLibrary);
workspaceTabs.forEach((tab) => tab.addEventListener("click", () => setWorkspace(tab.dataset.workspace)));
document.addEventListener("keydown", (event) => {
  if (event.target.matches("input, select, button")) return;
  if (event.code === "Space" && selectedVideo) { event.preventDefault(); playPauseButton.click(); }
  if (video.paused && event.key === "ArrowLeft") { event.preventDefault(); stepFrame(-1); }
  if (video.paused && event.key === "ArrowRight") { event.preventDefault(); stepFrame(1); }
});

window.SupervisorWorkspace = { openVideo, showLibrary };
setWorkspace("video");
