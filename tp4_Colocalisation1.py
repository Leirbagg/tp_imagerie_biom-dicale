import tifffile as tiff
import numpy as np
from scipy.stats import pearsonr
from skimage.filters import gaussian
from skimage.morphology import remove_small_objects
import matplotlib.pyplot as plt

# Charger les deux images (les deux canaux)
image1 = tiff.imread("/home/leirbag/Bureau/taff/M2/imagerie_biomed/PourEnvoi/Images/Coloc1.tif")
image2 = tiff.imread("/home/leirbag/Bureau/taff/M2/imagerie_biomed/PourEnvoi/Images/Coloc2.tif")

# Vérifier que les deux images ont les mêmes dimensions
if image1.shape != image2.shape:
    raise ValueError("Les deux images doivent avoir les mêmes dimensions.")

# Étape 1 : Réduction du bruit avec un filtre Gaussien
sigma = 2  # Ajustez le paramètre sigma selon le niveau de bruit
filtered_image1 = gaussian(image1, sigma=sigma)
filtered_image2 = gaussian(image2, sigma=sigma)

# Étape 2 : Seuillage pour supprimer les petits signaux (optionnel)
threshold1 = filtered_image1.mean() + filtered_image1.std()
threshold2 = filtered_image2.mean() + filtered_image2.std()

binary1 = filtered_image1 > threshold1
binary2 = filtered_image2 > threshold2

# Nettoyer les petits objets (résidus de bruit)
min_object_size = 50  # Ajustez cette valeur selon vos besoins
binary1_cleaned = remove_small_objects(binary1, min_size=min_object_size)
binary2_cleaned = remove_small_objects(binary2, min_size=min_object_size)

# Étape 3 : Appliquer les masques binaires pour enlever le bruit restant
cleaned_image1 = filtered_image1 * binary1_cleaned
cleaned_image2 = filtered_image2 * binary2_cleaned

# Étape 4 : Calcul du coefficient de Pearson
# Aplatir les images nettoyées en 1D pour le calcul de la corrélation
flattened_image1 = cleaned_image1.ravel()
flattened_image2 = cleaned_image2.ravel()

pearson_corr, _ = pearsonr(flattened_image1, flattened_image2)
print(f"Coefficient de corrélation de Pearson (après réduction du bruit) = {pearson_corr:.4f}")

# Étape 5 : Visualisation des résultats
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.title("Image 1 (bruitée)")
plt.imshow(image1, cmap='gray')

plt.subplot(2, 2, 2)
plt.title("Image 1 (après filtrage)")
plt.imshow(filtered_image1, cmap='gray')

plt.subplot(2, 2, 3)
plt.title("Image 2 (bruitée)")
plt.imshow(image2, cmap='gray')

plt.subplot(2, 2, 4)
plt.title("Image 2 (après filtrage)")
plt.imshow(filtered_image2, cmap='gray')

plt.tight_layout()
plt.show()
