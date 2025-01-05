import tifffile as tiff
import numpy as np
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects
from skimage.filters import gaussian
import trackpy as tp
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import measure
from scipy.ndimage import center_of_mass

tif_file = "/home/leirbag/Bureau/taff/M2/imagerie_biomed/Images_TP/cell2D_timelapse.tif"
image_stack = tiff.imread(tif_file)

print("Dimensions de l'image :", image_stack.shape)

smoothed_stack = np.array([gaussian(frame, sigma=2) for frame in image_stack])

segmented_stack = []

for frame in smoothed_stack:
    thresh = threshold_otsu(frame)
    binary_frame = frame > thresh
    binary_frame = remove_small_objects(binary_frame, min_size=50)
    segmented_stack.append(binary_frame)

segmented_stack = np.array(segmented_stack)

data = []
for frame_idx, frame in enumerate(segmented_stack):
    labeled_frame, _ = ndimage.label(frame)
    region_props = measure.regionprops(labeled_frame)
    
    for region in region_props:
        y, x = region.centroid
        data.append({"frame": frame_idx, "x": x, "y": y})

df = pd.DataFrame(data)

tp.link(df, search_range=5) 


cell_positions = [center_of_mass(frame) for frame in segmented_stack]
print("Centres de masse des cellules :", cell_positions)


import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots()

def update(frame_idx):
    ax.clear()
    ax.set_title(f"Frame {frame_idx}")
    
    ax.imshow(image_stack[frame_idx], cmap='gray')
    
    ax.contour(segmented_stack[frame_idx], colors='r')
    
    if cell_positions[frame_idx]:  
        ax.scatter(*cell_positions[frame_idx], color='yellow', label='Centre')
    
    ax.legend()

ani = FuncAnimation(fig, update, frames=len(image_stack), interval=500)

plt.show()
