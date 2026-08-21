const modeTabs = document.querySelectorAll(".mode-tab");
const modeLabel = document.querySelector("#mode-label");
const saveButton = document.querySelector("#save-button");
const saveStatus = document.querySelector("#save-status");
const saveStatusLabel = document.querySelector("#save-status-label");

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
