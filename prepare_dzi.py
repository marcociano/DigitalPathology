import os
import cv2
import pyvips
import openslide
import numpy as np



def convert_mrxs_to_dzi(input_mrxs, output_dir):
    """Converte un file .mrxs direttamente in formato DZI utilizzando OpenSlide e PyVips."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    base_name = os.path.splitext(os.path.basename(input_mrxs))[0]
    dzi_path = os.path.join(output_dir, f"{base_name}.dzi")

    try:
        print(f"Convertendo '{input_mrxs}' direttamente in .dzi...")

        # Usa OpenSlide per leggere il file
        slide = openslide.OpenSlide(input_mrxs)

        # Converte direttamente in Deep Zoom Image
        pyvips.Image.new_from_file(input_mrxs, access="sequential").dzsave(
            os.path.join(output_dir, base_name),
            tile_size=256,
            suffix=".jpg",
            depth="onepixel"
        )

        print(f"File .dzi generato in: {dzi_path}")
        return dzi_path
    except Exception as e:
        print(f"Errore nella conversione in DZI da .mrxs: {e}")
        return None


def convert_tiff_to_dzi(input_tiff, output_dir):
    """Converte un file .tiff in formato DZI."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    base_name = os.path.splitext(os.path.basename(input_tiff))[0]
    dzi_path = os.path.join(output_dir, f"{base_name}.dzi")

    try:
        print(f"Convertendo '{input_tiff}' in .dzi...")

        pyvips.Image.new_from_file(input_tiff, access="sequential").dzsave(
            os.path.join(output_dir, base_name),
            tile_size=256,
            suffix=".jpg",
            depth="onepixel"
        )

        print(f"File .dzi generato in: {dzi_path}")
        return dzi_path
    except Exception as e:
        print(f"Errore nella conversione in DZI: {e}")
        return None


def generate_edge_image(input_path, output_path):
    """Funzione per l'estrazione degli edges dell'immagine processata"""
    try:
        slide = openslide.OpenSlide(input_path)
        width, height = slide.dimensions

        region = slide.read_region((0, 0), 0, (width, height))
        region = region.convert("L")
        image = np.array(region)

        # Applica Canny per l'edge detection
        edges = cv2.Canny(image, 50, 150)

        edges_pil = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        cv2.imwrite(output_path, edges_pil)

        print(f"Edge detection completata e salvata in: {output_path}")

    except MemoryError:
        print("Errore: Memoria insufficiente per elaborare gli edges.")

    except Exception as e:
        print(f"Errore nella generazione dell'immagine edge: {e}")


def generate_edge_image_thumbnail(input_path, output_path):
    """Genera edges da una miniatura invece che dall'immagine completa"""
    try:
        slide = openslide.OpenSlide(input_path)
        thumbnail = slide.get_thumbnail((2048, 2048))  # Usa solo una miniatura 1024x1024
        image = np.array(thumbnail.convert("L"))

        # Edge detection
        edges = cv2.Canny(image, 50, 150)
        edges_pil = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        cv2.imwrite(output_path, edges_pil)

        print(f"Edge detection (da miniatura) completata in: {output_path}")

    except Exception as e:
        print(f"Errore nella generazione dell'immagine edge: {e}")



def main():
    input_image = "scanned_Images/Mirax2.2-4-PNG.mrxs"
    edge_output_image = "scanned_Images/Mirax2.2-4-PNG_edges.tiff"
    dzi_output_folder = "dzi_output"


    if input_image.lower().endswith(".mrxs"):
        print(f"Rilevato file .mrxs: {input_image}")
        convert_mrxs_to_dzi(input_image, dzi_output_folder)
        generate_edge_image_thumbnail(input_image, edge_output_image)
        convert_tiff_to_dzi(edge_output_image, dzi_output_folder)
    elif input_image.lower().endswith(".tiff") or input_image.lower().endswith(".tif"):
        print(f"Rilevato file .tiff: {input_image}")
        convert_tiff_to_dzi(input_image, dzi_output_folder)
        generate_edge_image(input_image, edge_output_image)
        convert_tiff_to_dzi(edge_output_image, dzi_output_folder)


if __name__ == "__main__":
    main()
