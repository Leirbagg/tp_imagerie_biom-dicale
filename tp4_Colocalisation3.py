import tifffile as tiff
import numpy as np
from skimage.filters import gaussian
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

image_stack1 = tiff.imread("/home/leirbag/Bureau/taff/M2/imagerie_biomed/Images_TP/ParticuleTracking01.tif")  # Canal 1
image_stack2 = tiff.imread("/home/leirbag/Bureau/taff/M2/imagerie_biomed/Images_TP/ParticuleTracking02.tif")  # Canal 1

if image_stack1.shape != image_stack2.shape:
    raise ValueError("Les deux séquences doivent avoir les mêmes dimensions.")

num_frames = image_stack1.shape[0]
colocalization_percentages = []
mean_colocalization_distances = []

# Paramètres
sigma = 2  # Paramètre du filtre Gaussien pour réduire le bruit
min_object_size = 50  # Taille minimale des objets (en pixels)
max_distance = 20  # Distance maximale pour considérer deux objets comme colocalisés

for frame_idx in range(num_frames):
    print(f"Traitement de la frame {frame_idx + 1}/{num_frames}...")

    frame1 = image_stack1[frame_idx]
    frame2 = image_stack2[frame_idx]

    filtered_frame1 = gaussian(frame1, sigma=sigma)
    filtered_frame2 = gaussian(frame2, sigma=sigma)

    threshold1 = filtered_frame1.mean() + filtered_frame1.std()
    threshold2 = filtered_frame2.mean() + filtered_frame2.std()

    binary1 = filtered_frame1 > threshold1
    binary2 = filtered_frame2 > threshold2

    binary1_cleaned = remove_small_objects(binary1, min_size=min_object_size)
    binary2_cleaned = remove_small_objects(binary2, min_size=min_object_size)

    labeled1 = label(binary1_cleaned)
    labeled2 = label(binary2_cleaned)

    objects1 = [region.centroid for region in regionprops(labeled1)]
    objects2 = [region.centroid for region in regionprops(labeled2)]

    if len(objects1) == 0 or len(objects2) == 0:
        print(f"Aucun objet détecté dans la frame {frame_idx + 1}.")
        colocalization_percentages.append(0)
        mean_colocalization_distances.append(np.nan)
        continue

    tree1 = cKDTree(objects1)
    tree2 = cKDTree(objects2)

    distances1, indices1 = tree1.query(objects2, distance_upper_bound=max_distance)
    distances2, indices2 = tree2.query(objects1, distance_upper_bound=max_distance)

    colocalized_indices1 = np.where(distances1 < max_distance)[0]
    colocalized_indices2 = np.where(distances2 < max_distance)[0]

    colocalized_count = len(colocalized_indices1)
    percentage_colocalized = (colocalized_count / max(len(objects1), len(objects2))) * 100
    colocalization_percentages.append(percentage_colocalized)

    if colocalized_count > 0:
        mean_distance = np.mean(distances1[colocalized_indices1])
    else:
        mean_distance = np.nan
    mean_colocalization_distances.append(mean_distance)

    print(f"Frame {frame_idx + 1}: Pourcentage de colocalisation = {percentage_colocalized:.2f}%, Distance moyenne = {mean_distance:.2f} pixels")

print("\nRésumé des colocalisations :")
for idx, (percentage, distance) in enumerate(zip(colocalization_percentages, mean_colocalization_distances)):
    print(f"Frame {idx + 1}: {percentage:.2f}% colocalisé, distance moyenne = {distance:.2f} pixels")

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(range(1, num_frames + 1), colocalization_percentages, marker='o')
plt.title("Pourcentage de colocalisation par frame")
plt.xlabel("Frame")
plt.ylabel("Pourcentage de colocalisation (%)")
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(range(1, num_frames + 1), mean_colocalization_distances, marker='o')
plt.title("Distance moyenne de colocalisation par frame")
plt.xlabel("Frame")
plt.ylabel("Distance moyenne (pixels)")
plt.grid()

plt.tight_layout()
plt.show()
