"use strict";

/* ===== Responsive Sidepanel ====== */
const sidePanelToggler = document.getElementById("sidepanel-toggler");
const sidePanel = document.getElementById("app-sidepanel");
const sidePanelDrop = document.getElementById("sidepanel-drop");
const sidePanelClose = document.getElementById("sidepanel-close");

window.addEventListener("load", function () {
  responsiveSidePanel();
});

window.addEventListener("resize", function () {
  responsiveSidePanel();
});

function responsiveSidePanel() {
  let w = window.innerWidth;
  if (w >= 1200) {
    // if larger
    //console.log('larger');
    sidePanel.classList.remove("sidepanel-hidden");
    sidePanel.classList.add("sidepanel-visible");
  } else {
    // if smaller
    //console.log('smaller');
    sidePanel.classList.remove("sidepanel-visible");
    sidePanel.classList.add("sidepanel-hidden");
  }
}

sidePanelToggler.addEventListener("click", () => {
  if (sidePanel.classList.contains("sidepanel-visible")) {
    console.log("visible");
    sidePanel.classList.remove("sidepanel-visible");
    sidePanel.classList.add("sidepanel-hidden");
  } else {
    console.log("hidden");
    sidePanel.classList.remove("sidepanel-hidden");
    sidePanel.classList.add("sidepanel-visible");
  }
});

sidePanelClose.addEventListener("click", (e) => {
  e.preventDefault();
  sidePanelToggler.click();
});

sidePanelDrop.addEventListener("click", (e) => {
  sidePanelToggler.click();
});
