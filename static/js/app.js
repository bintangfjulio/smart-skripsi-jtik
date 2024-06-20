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

window.chartColors = {
  green: "#75c181",
  blue: "#5b99ea",
  gray: "#a9b5c9",
  yellow: "#f1c40f",
  text: "#252930",
  border: "#e7e9ed",
};

function initChart(keys, values) {
  var doughnutChartConfig = {
    type: "doughnut",
    data: {
      datasets: [
        {
          data: values,
          backgroundColor: [
            window.chartColors.green,
            window.chartColors.blue,
            window.chartColors.grey,
            window.chartColors.yellow,
          ],
          label: "Hasil",
        },
      ],
      labels: keys,
    },
    options: {
      responsive: true,
      legend: {
        display: true,
        position: "bottom",
        align: "center",
      },

      tooltips: {
        titleMarginBottom: 10,
        bodySpacing: 10,
        xPadding: 16,
        yPadding: 16,
        borderColor: window.chartColors.border,
        borderWidth: 1,
        backgroundColor: "#fff",
        bodyFontColor: window.chartColors.text,
        titleFontColor: window.chartColors.text,

        animation: {
          animateScale: true,
          animateRotate: true,
        },

        callbacks: {
          label: function (tooltipItem, data) {
            var dataset = data.datasets[tooltipItem.datasetIndex];
            var total = dataset.data.reduce(function (
              previousValue,
              currentValue,
              currentIndex,
              array
            ) {
              return previousValue + currentValue;
            });

            var currentValue = dataset.data[tooltipItem.index];
            var percentage = Math.floor((currentValue / total) * 100 + 0.5);

            return percentage + "%";
          },
        },
      },
    },
  };

  return doughnutChartConfig;
}
