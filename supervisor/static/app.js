const modeTabs = document.querySelectorAll(".mode-tab");
const modeLabel = document.querySelector("#mode-label");
const saveButton = document.querySelector("#save-button");
const saveStatus = document.querySelector("#save-status");
const saveStatusLabel = document.querySelector("#save-status-label");
const video = document.querySelector("#video");
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

let selectedVideo = null;

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
});

video.addEventListener("timeupdate", updatePlaybackDisplay);
video.addEventListener("seeked", updatePlaybackDisplay);
video.addEventListener("play", () => {
  playPauseButton.textContent = "❚❚";
  playPauseButton.setAttribute("aria-label", "Pause");
});
video.addEventListener("pause", () => {
  playPauseButton.textContent = "▶";
  playPauseButton.setAttribute("aria-label", "Play");
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
