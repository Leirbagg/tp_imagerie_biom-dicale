import numpy as np
from tifffile import imread
import cv2
from scipy.spatial import distance
from skimage.feature import blob_log
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt

def analyze_tif(file_path):
    images = imread(file_path)
    num_frames, height, width = images.shape
    
    blob_params = {
        'min_sigma': 4, 'max_sigma': 7, 'num_sigma': 10, 'threshold': 0.02
    }

    trajectories = [] 

    for i in range(num_frames):
        frame = images[i]
        blobs = blob_log(frame, **blob_params)

        positions = [(int(blob[1]), int(blob[0])) for blob in blobs] 

        if i == 0:
            trajectories = [[pos] for pos in positions]
        else:
            last_positions = [traj[-1] for traj in trajectories]
            dist_matrix = distance.cdist(last_positions, positions)

            max_dist = 2  # Set a maximum distance for particle tracking
            for t, traj in enumerate(trajectories):
                min_dist_index = np.argmin(dist_matrix[t])
                if dist_matrix[t][min_dist_index] < max_dist:
                    traj.append(positions[min_dist_index])  
                else:
                    traj.append(traj[-1])  

    trajectory_lengths = []
    for traj in trajectories:
        traj_len = sum(
            distance.euclidean(traj[i], traj[i + 1]) for i in range(len(traj) - 1)
        )
        trajectory_lengths.append(traj_len)
    avg_trajectory_length = np.mean(trajectory_lengths) if trajectory_lengths else 0

    info = {
        'Resolution (Height, Width)': (height, width),
        'Number of Frames': num_frames,
        'Average Trajectory Length': avg_trajectory_length,
        'Number of Particles': len(trajectories),
    }
    
    return info, trajectories

file_path = 'Images_TP/ParticuleTracking01.tif'
info, trajectories = analyze_tif(file_path)

print("Sequence Information:")
for key, value in info.items():
    print(f"{key}: {value}")

for traj in trajectories[:271]: 
    plt.plot([p[0] for p in traj], [p[1] for p in traj])
plt.title("Sample Particle Trajectories")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.show()
