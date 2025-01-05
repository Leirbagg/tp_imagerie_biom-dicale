import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

image_path = "/home/leirbag/Bureau/taff/M2/imagerie_biomed/Images_TP/101878272.png"
image = cv2.imread(image_path)

image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

lower_brown = np.array([10, 50, 20])  # Limite basse pour le marron
upper_brown = np.array([30, 255, 200])  # Limite haute pour le marron

mask = cv2.inRange(image_hsv, lower_brown, upper_brown)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

labeled_mask, num_objects = ndimage.label(mask_cleaned)

print(f"Nombre de spots marrons détectés : {num_objects}")

plt.figure(figsize=(15, 10))

plt.subplot(1, 3, 1)
plt.title("Image originale")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

plt.subplot(1, 3, 2)
plt.title("Masque des spots marrons")
plt.imshow(mask, cmap='gray')

plt.subplot(1, 3, 3)
plt.title("Image avec annotations")
annotated_image = image.copy()
for spot_idx in range(1, num_objects + 1):
    coords = np.argwhere(labeled_mask == spot_idx)
    y, x = coords.mean(axis=0).astype(int)
    cv2.circle(annotated_image, (x, y), 10, (0, 255, 0), 2)
    cv2.putText(annotated_image, f"{spot_idx}", (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))

plt.tight_layout()
plt.show()

