document.getElementById("screenshot-button").addEventListener("click", () => {
    const fileNameInput = document.getElementById("screenshot-name");
    const fileName = fileNameInput.value || "screenshot";

    const viewport = viewer.viewport;
    const region = viewport.viewportToImageRectangle(viewport.getBounds());

    const requestData = {
        file_name: fileName,
        region_x: Math.round(region.x),
        region_y: Math.round(region.y),
        width: Math.round(region.width),
        height: Math.round(region.height),
    };

    fetch("/screenshot", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestData),
    })
        .then((response) => response.json())
        .then((data) => {
            const messageElement = document.getElementById("screenshot-message");
            if (data.status === "success") {
                messageElement.textContent = data.message;
                messageElement.style.color = "green";
            } else {
                messageElement.textContent = data.message;
                messageElement.style.color = "red";
            }
        })
        .catch((error) => {
            console.error("Errore durante lo screenshot:", error);
        });

});
