// Soften pydata-sphinx-theme's default sidebar headings for a more docs-like feel.
// Edit the strings below to tweak labels.
(function () {
  var LEFT_LABEL = "Contents";   // replaces "Section Navigation"
  var RIGHT_LABEL = "On this page"; // replaces "On this page"

  function relabel() {
    document
      .querySelectorAll(".bd-docs-nav .bd-links__title")
      .forEach(function (el) {
        el.textContent = LEFT_LABEL;
      });

    document
      .querySelectorAll(".tocsection.onthispage")
      .forEach(function (el) {
        el.childNodes.forEach(function (node) {
          if (
            node.nodeType === Node.TEXT_NODE &&
            node.textContent.trim() === "On this page"
          ) {
            node.textContent = " " + RIGHT_LABEL;
          }
        });
      });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", relabel);
  } else {
    relabel();
  }
})();
