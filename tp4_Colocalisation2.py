import tifffile as tiff
import numpy as np
from skimage.filters import gaussian
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

image1 = tiff.imread("/home/leirbag/Bureau/taff/M2/imagerie_biomed/PourEnvoi/Images/Coloc1.tif")
image2 = tiff.imread("/home/leirbag/Bureau/taff/M2/imagerie_biomed/PourEnvoi/Images/Coloc2.tif")

if image1.shape != image2.shape:
    raise ValueError("Les deux images doivent avoir les mêmes dimensions.")

sigma = 2  # Ajustez la valeur de sigma selon le niveau de bruit
filtered_image1 = gaussian(image1, sigma=sigma)
filtered_image2 = gaussian(image2, sigma=sigma)

threshold1 = filtered_image1.mean() + filtered_image1.std()  # Ajustez si nécessaire
threshold2 = filtered_image2.mean() + filtered_image2.std()

binary1 = filtered_image1 > threshold1
binary2 = filtered_image2 > threshold2

labeled1 = label(binary1)
labeled2 = label(binary2)

objects1 = [region.centroid for region in regionprops(labeled1)]
objects2 = [region.centroid for region in regionprops(labeled2)]

if len(objects1) == 0 or len(objects2) == 0:
    print("Aucun objet détecté dans l'une des images.")
    exit()

tree1 = cKDTree(objects1)
tree2 = cKDTree(objects2)

# Distance maximale pour considérer deux objets comme colocalisés
max_distance = 20  # Ajustez selon vos besoins

distances1, indices1 = tree1.query(objects2, distance_upper_bound=max_distance)
distances2, indices2 = tree2.query(objects1, distance_upper_bound=max_distance)

colocalized_indices1 = np.where(distances1 < max_distance)[0]
colocalized_indices2 = np.where(distances2 < max_distance)[0]

colocalized_count = len(colocalized_indices1)
percentage_colocalized = (colocalized_count / max(len(objects1), len(objects2))) * 100

if colocalized_count > 0:
    mean_distance = np.mean(distances1[colocalized_indices1])
else:
    mean_distance = np.nan

print(f"Pourcentage de colocalisation : {percentage_colocalized:.2f}%")
print(f"Distance moyenne de colocalisation : {mean_distance:.2f} pixels")

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title("Image 1 avec objets détectés")
plt.imshow(image1, cmap='gray')
for y, x in objects1:
    plt.scatter(x, y, color='red', s=10)

plt.subplot(1, 2, 2)
plt.title("Image 2 avec objets détectés")
plt.imshow(image2, cmap='gray')
for y, x in objects2:
    plt.scatter(x, y, color='blue', s=10)

plt.tight_layout()
plt.show()
