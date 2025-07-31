
#Demo appliying padding 
# demo_apply_circular_mask.py
# ----------------------------------------------------------
# Demo: Reducing Truncation Artifacts Using Edge Padding
#
# This demo shows how padding the detector width and updating 
# geometry accordingly can suppress truncation artifacts in FDK reconstruction.
#
# Workflow:
# 1. Initialize TIGRE geometry (high resolution)
# 2. Load sample phantom (head)
# 3. Simulate cone-beam projections from the phantom
# 4. Define padded geometry (25% detector width padding)
# 5. Pad projections accordingly
# 6. Reconstruct both padded and unpadded projections using FDK
# 7. Visualize and compare results
#
# Coded by:     Kubra Kumrular (RKK)
# ----------------------------------------------------------

#% Imports
import      tigre
import      numpy as np
from        tigre.utilities import sample_loader
from        tigre.utilities import CTnoise
import      tigre.algorithms as algs
import      matplotlib.pyplot as plt
from        tigre.utilities import gpu
import      copy
gpu_names   = gpu.getGpuNames()        # 
gpuids      = gpu.getGpuIds(gpu_names[0])  #  

#%% Geometry

geo         = tigre.geometry_default(high_resolution=True)

#%% Load data and generate projections

# define angles
angles      = np.linspace(0, 2 * np.pi, 512)
# Load thorax phantom data
head        = sample_loader.load_head_phantom(geo.nVoxel)
# generate projections
projections = tigre.Ax(head, geo, angles)
print(geo)
print(projections.shape)

#%% --- Define 3D  paddded geometry ---

# Geometry for padding
pad_width                  = int(round((geo.nDetector[1]) * 0.25)) # 25 percent padding to both sides of your detector (in X) 
geo_padded                 = copy.deepcopy(geo)
geo_padded.nDetector[1]    += 2 * pad_width
geo_padded.sDetector[1]    = geo_padded.nDetector[1] * geo_padded.dDetector[1]
geo_padded.nVoxel[1:]      = geo_padded.nDetector[1]
geo_padded.sVoxel[1:]      = geo_padded.nVoxel[1] * geo.dVoxel[1]
geo_padded.dVoxel          = geo_padded.sVoxel / geo_padded.nVoxel

#print(geo_padded)

# Pad only width (X axis- U axis) of projections 
projections_padded = np.pad(projections, ((0, 0), (0, 0), (pad_width, pad_width)), mode='edge')

# Print shapes to compare original and padded projection dimensions
#print("Original projections shape: ", projections.shape)
#print("Padded projections shape:   ", projections_padded.shape)

#%% --- Visualize ---
#%
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.imshow(projections[0])
plt.title("No Padding")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(projections_padded[0])
plt.title(" Padded projections (edge)")
plt.axis("off")
plt.show()


#%% --- Reconstruction---

# Reconstruction no padding
imgFDK          = algs.fdk(projections , geo, angles, gpuids=gpuids)

# Reconstruction with padding
imgFDK_with_pad = algs.fdk(projections_padded , geo_padded, angles, gpuids=gpuids)


#%% --- Visualize ---
#%
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.imshow(imgFDK[0], cmap='viridis', vmin=0, vmax=0.1)
plt.title("No Padding")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(imgFDK_with_pad[0, pad_width:-pad_width, pad_width:-pad_width],cmap='viridis',vmin=0, vmax=0.1)
plt.title(" With Padding")
plt.axis("off")
plt.show()

#%%
#% --- Visualize ---
#plt.figure(figsize=(18,5))
#plt.subplot(1,3,1)
#plt.imshow(imgFDK[0], cmap='viridis', vmin=0, vmax= 0.1)
#plt.title("No Padding")
#plt.axis("off")

#plt.subplot(1,3,2)
#plt.imshow(imgFDK_with_pad[0], cmap='viridis', vmin=0, vmax= 0.1)
#plt.title(" With Padding (edge)")
#plt.axis("off")

#plt.subplot(1,3,3)
#plt.imshow(imgFDK_with_pad[0, pad_width:-pad_width, pad_width:-pad_width],cmap='viridis', vmin=0, vmax= 0.1)
#plt.title(" With Padding (edge)")
#plt.axis("off")
#plt.show()
# %%
