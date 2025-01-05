import cv2
import numpy as np
import matplotlib.pyplot as plt

image1 = cv2.imread("/home/leirbag/Bureau/taff/M2/imagerie_biomed/Images_TP/he.png")
image2 = cv2.imread("/home/leirbag/Bureau/taff/M2/imagerie_biomed/Images_TP/hdAB.png")

image1_hsv = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)
image2_hsv = cv2.cvtColor(image2, cv2.COLOR_BGR2HSV)

lower_blue = np.array([90, 50, 50])  # Limite basse pour le bleu
upper_blue = np.array([130, 255, 255])  # Limite haute pour le bleu

mask1 = cv2.inRange(image1_hsv, lower_blue, upper_blue)
mask2 = cv2.inRange(image2_hsv, lower_blue, upper_blue)

result1 = cv2.bitwise_and(image1, image1, mask=mask1)
result2 = cv2.bitwise_and(image2, image2, mask=mask2)

plt.figure(figsize=(10, 10))

plt.subplot(2, 2, 1)
plt.title("Image 1 Originale")
plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))

plt.subplot(2, 2, 2)
plt.title("Image 1 - Zones Bleues")
plt.imshow(cv2.cvtColor(result1, cv2.COLOR_BGR2RGB))

plt.subplot(2, 2, 3)
plt.title("Image 2 Originale")
plt.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))

plt.subplot(2, 2, 4)
plt.title("Image 2 - Zones Bleues")
plt.imshow(cv2.cvtColor(result2, cv2.COLOR_BGR2RGB))

plt.tight_layout()
plt.show()