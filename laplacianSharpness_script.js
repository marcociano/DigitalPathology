function updateSharpness() {
    const zoomFactor = viewer.viewport.getZoom();
    const level = Math.round(zoomFactor * maxLevel);

    fetch("/sharpness", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ level: level, zoom_factor: zoomFactor }),
    })
        .then(response => response.json())
        .then(data => {
            const sharpnessStatus = document.getElementById("sharpness-status");
            if (data.variance !== null) {
                sharpnessStatus.innerText = `
                    ${data.status}
                    (Variance: ${data.variance.toFixed(2)},
                    Threshold: ${data.threshold.toFixed(2)},
                    Ratio: ${data.sharpness_ratio.toFixed(2)})`;
            } else {
                sharpnessStatus.innerText = "Error calculating sharpness";
            }
        })
        .catch(err => {
            console.error("Errore nel calcolo della nitidezza:", err);
        });
}
