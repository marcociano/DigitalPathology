import os
import cv2
import numpy as np
import openslide
from flask import Flask, render_template, send_from_directory, request, jsonify

app = Flask(__name__)

# Percorso alla directory delle tiles
TILE_DIR = os.path.abspath("dzi_output")
FILE_PATH = "scanned_Images/Mirax2.2-4-PNG.mrxs"
SCREENSHOT_DIR = os.path.abspath("screenshots")

if not os.path.exists(SCREENSHOT_DIR):
    os.makedirs(SCREENSHOT_DIR)


@app.route("/get_microns_per_pixel")
def get_microns_per_pixel():
    try:
        slide = openslide.OpenSlide(FILE_PATH)
        microns_per_pixel_x = float(slide.properties.get("openslide.mpp-x", "0"))
        microns_per_pixel_y = float(slide.properties.get("openslide.mpp-y", "0"))
        width, height = slide.dimensions

        return jsonify({
            "width": width,
            "height": height,
            "microns_per_pixel_x": microns_per_pixel_x,
            "microns_per_pixel_y": microns_per_pixel_y
        })
    except Exception as e:
        print(f"Errore nel recupero dei metadati dell'immagine: {e}")
        return jsonify({"error": "Impossibile recuperare i metadati"}), 500


@app.route("/")
def index():
    file_path = FILE_PATH

    slide = openslide.OpenSlide(file_path)
    max_level = len(slide.level_dimensions) - 1  # Numero massimo di livelli disponibili

    return render_template("index.html", max_level=max_level)


@app.route("/sharpness", methods=["POST"])
def sharpness():
    data = request.get_json()
    level = int(data.get("level", 0))
    zoom_factor = data.get("zoom_factor", 1.0)

    # Threshold dinamica con un fattore di scaling
    base_threshold = 65
    scaling_coefficient = -0.25
    dynamic_threshold = base_threshold * (zoom_factor ** scaling_coefficient)

    try:
        slide = openslide.OpenSlide(FILE_PATH)
        max_levels = len(slide.level_dimensions) - 1

        if level > max_levels:
            level = max_levels
        elif level < 0:
            level = 0

        # Ottieni dimensioni del livello
        width, height = slide.level_dimensions[level]
        region_size = min(1024, width, height)
        region_x = max(0, (width - region_size) // 2)
        region_y = max(0, (height - region_size) // 2)

        # Estrai la regione
        region = slide.read_region((region_x, region_y), level, (region_size, region_size))
        region = region.convert("L")
        image = np.array(region)

        # Calcola la varianza di Laplace
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        variance = laplacian.var()

        # Normalizza la varianza in base alle dimensioni
        normalized_variance = variance / (width * height)

        # Percentuale rispetto alla threshold
        sharpness_ratio = normalized_variance / dynamic_threshold

        # Log per debug
        print(f"Zoom Factor: {zoom_factor}")
        print(f"Level: {level}, Width: {width}, Height: {height}")
        print(f"Variance: {variance}, Normalized Variance: {normalized_variance}")
        print(f"Threshold: {dynamic_threshold}, Sharpness Ratio: {sharpness_ratio}")

        # Determina il risultato basato sulla ratio
        is_blurry = not (0.8 <= sharpness_ratio <= 1.2)

        return jsonify({
            "variance": variance,
            "normalized_variance": normalized_variance,
            "threshold": dynamic_threshold,
            "sharpness_ratio": sharpness_ratio,
            "status": "Blurry" if is_blurry else "Sharp"
        })
    except openslide.OpenSlideError as e:
        return jsonify({"variance": None, "status": "Error", "message": "Impossibile elaborare l'immagine al livello selezionato."})
    except Exception as e:
        print(f"Errore durante il calcolo della nitidezza: {e}")
        return jsonify({"variance": None, "status": "Error", "message": str(e)})


@app.route("/screenshot", methods=["POST"])
def screenshot():
    data = request.get_json()
    screenshot_name = data.get("name", "screenshot.png")

    if not screenshot_name.lower().endswith(".png"):
        screenshot_name += ".png"

    file_path = os.path.join(SCREENSHOT_DIR, screenshot_name)

    try:
        # Ottenimento dimensioni immagine
        slide = openslide.OpenSlide(FILE_PATH)
        region = slide.get_thumbnail((1024, 1024))
        region.save(file_path)
        return jsonify({"status": "success", "message": f"Screenshot salvato come {screenshot_name}"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route("/dzi_output/<path:filename>")
def tiles_file(filename):
    return send_from_directory(TILE_DIR, filename)

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)

