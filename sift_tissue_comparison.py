import cv2
import numpy as np


def preprocess_image(img):
    """Preprocessing dell'immagine per migliorare il SIFT matching i differenti colori dei tessuti (vetrino e paraffina)"""
    # Conversione in scala di grigi
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Applicazione della procedura di equalizzazione all'istogramma per normalizzare le distribuzioni di intensità
    equalized = cv2.equalizeHist(gray)

    # Applicazione di CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(equalized)

    # Applicazione del filtro bilaterale per ridurre il rumore e preservare i contorni
    filtered = cv2.bilateralFilter(enhanced, 9, 75, 75)

    return filtered


def create_tissue_mask(img):
    """Creazione di una maschera per identificare le regioni del tessuto ed escludere parte vitree/paraffina"""
    # Conversione in scala di grigi
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Viene utilizzata la soglia di Otsu per separare l'oggetto dallo sfondo (tessuto/vetrino o tessuto/paraffina)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Inverte se necessario il 'colore' tra lo sfondo e l'oggetto
    if np.mean(binary) > 127:
        binary = cv2.bitwise_not(binary)

    # Applicazione di operazioni morfologiche per rendere la maschera più pulita
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Rimozione di piccole regioni che presentano rumore
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) < 2000:  # Remove small areas
            cv2.fillPoly(mask, [contour], 0)

    return mask


def filter_keypoints_by_mask(keypoints, descriptors, mask):
    """Filtering dei keypoints per 'selezionare' solo quelli all'interno del tessuto"""
    if descriptors is None:
        return keypoints, descriptors

    filtered_kp = []
    filtered_desc = []

    for i, kp in enumerate(keypoints):
        x, y = int(kp.pt[0]), int(kp.pt[1])
        if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
            if mask[y, x] > 0:  # Point is in tissue region
                filtered_kp.append(kp)
                filtered_desc.append(descriptors[i])

    if len(filtered_desc) > 0:
        return filtered_kp, np.array(filtered_desc)
    else:
        return [], None


def evaluate_tissue_similarity(num_matches, kp1_count, kp2_count):
    """Valutazione basata sul numero di matchs e keypoint"""

    if kp1_count == 0 or kp2_count == 0:
        return False, "Nessun keypoint trovato sui tessuti"

    min_keypoints = min(kp1_count, kp2_count)
    max_keypoints = max(kp1_count, kp2_count)
    match_ratio = num_matches / min_keypoints if min_keypoints > 0 else 0

    # Controllo dei Keypoints con numero inferiore a 50
    if min_keypoints < 50:
        if num_matches >= 30 and match_ratio >= 0.6:
            return True, f"TESSUTI PROBABILMENTE IDENTICI - {num_matches} matches, ratio: {match_ratio:.2%}"
        else:
            return False, f"TESSUTI DIVERSI - Troppo pochi keypoints ({min_keypoints}) per confronto affidabile, {num_matches} matches"


    if num_matches >= 100 and match_ratio >= 0.25:
        return True, f"TESSUTI IDENTICI - {num_matches} matches, ratio: {match_ratio:.2%}"
    elif num_matches >= 50 and match_ratio >= 0.15:
        return True, f"TESSUTI PROBABILMENTE IDENTICI - {num_matches} matches, ratio: {match_ratio:.2%}"
    elif num_matches >= 25 and match_ratio >= 0.08:
        return False, f"TESSUTI POSSIBILMENTE SIMILI ma probabilmente diversi - {num_matches} matches, ratio: {match_ratio:.2%}"
    else:
        return False, f"TESSUTI DIVERSI - Solo {num_matches} matches, ratio: {match_ratio:.2%}"


vetrine = cv2.imread("Images_for_SIFT/vetrine1.jpg")
block = cv2.imread("Images_for_SIFT/tissue_block1.jpg")


if vetrine.shape == block.shape:
    print("Le immagini hanno la stessa dimensione e canali")
    difference = cv2.subtract(vetrine, block)
    b, g, r = cv2.split(difference)

    if cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0:
        print("Le due immagini sono completamente uguali")
    else:
        print("Le immagini sono diverse")
else:
    print("Le immagini hanno dimensioni diverse")

print("Creazione maschere del tessuto...")
mask1 = create_tissue_mask(vetrine)
mask2 = create_tissue_mask(block)

vetrine_processed = preprocess_image(vetrine)
block_processed = preprocess_image(block)

sift = cv2.SIFT_create(nfeatures=1000, contrastThreshold=0.03, edgeThreshold=15)

kp_1, desc_1 = sift.detectAndCompute(vetrine_processed, None)
kp_2, desc_2 = sift.detectAndCompute(block_processed, None)

if desc_1 is None or desc_2 is None:
    print("Errore: Impossibile estrarre descriptors da una o entrambe le immagini")
    exit()

print(f"Keypoints totali - Immagine 1: {len(kp_1)}, Immagine 2: {len(kp_2)}")

kp_1_filtered, desc_1_filtered = filter_keypoints_by_mask(kp_1, desc_1, mask1)
kp_2_filtered, desc_2_filtered = filter_keypoints_by_mask(kp_2, desc_2, mask2)

if desc_1_filtered is None or desc_2_filtered is None:
    print("Errore: Nessun keypoint trovato sui tessuti dopo il filtraggio")
    exit()

print(f"Keypoints sui tessuti - Immagine 1: {len(kp_1_filtered)}, Immagine 2: {len(kp_2_filtered)}")

index_params = dict(algorithm=0, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(desc_1_filtered, desc_2_filtered, k=2)

good_points = []
for match_pair in matches:
    if len(match_pair) == 2:
        m, n = match_pair
        if m.distance < 0.8 * n.distance:
            good_points.append(m)

print(f"Matches validi trovati: {len(good_points)}")

is_same, evaluation_message = evaluate_tissue_similarity(
    len(good_points), len(kp_1_filtered), len(kp_2_filtered)
)

print("\n" + "=" * 60)
print("RISULTATO COMPARAZIONE TESSUTI")
print("=" * 60)
print(evaluation_message)
print("=" * 60)

if len(good_points) > 0:
    result = cv2.drawMatches(vetrine, kp_1_filtered, block, kp_2_filtered, good_points, None)

    cv2.imshow('SIFT Matches', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Nessun match valido trovato tra i tessuti")
