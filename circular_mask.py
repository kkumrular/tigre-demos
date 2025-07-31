
# demo_apply_circular_mask.py
# ----------------------------------------------------------
# Demo for: Applying a Circular Mask on a Reconstructed Slice
#
# This demo:
# This demo shows how to apply a smooth, circular tapering mask
# to reconstructed volumes in order to reduce noise and edge effects.

# Workflow:
# 1. Initialize TIGRE cone-beam geometry
# 2. Load a phantom object (head)
# 3. Simulate cone-beam projections from the phantom
# 4. Reconstruct volume using FDK algorithm
# 5. Apply a smooth circular mask to the reconstructed volume
# 6. Visualize and compare original vs masked slices
#
# Author: Kubra Kumrular (RKK)
# ----------------------------------------------------------

# Imports
import      tigre
import      numpy as np
from        tigre.utilities import sample_loader
import      tigre.algorithms as algs
import      matplotlib.pyplot as plt
from        tigre.utilities import gpu
from        joblib import Parallel, delayed
import      multiprocessing
gpu_names = gpu.getGpuNames()        # 
gpuids = gpu.getGpuIds(gpu_names[0])  #  

#%% Geometry

geo = tigre.geometry_default(high_resolution=True)

#%% Load data and generate projections

# define angles
angles = np.linspace(0, 2 * np.pi, 512)

# Load thorax phantom data
head = sample_loader.load_head_phantom(geo.nVoxel)

# generate projections
projections = tigre.Ax(head, geo, angles)
#print(geo)

#%% --- Reconstruction---

imgFDK = algs.fdk(projections, geo, angles)

#%% Apply a smooth circular mask to the reconstructed volume 

# define circular mask
def apply_circular_mask(recon, geo, radius=0.99):
    """
    Applies a smooth circular mask to each reconstructed slice 
    to suppress noise and artifacts near the edges(for TIGRE recon).
    
    Parameters:
    -----------
    recon : numpy.ndarray
        3D reconstructed volume (shape: [nz, ny, nx]).
    geo : TIGRE geometry object
        Contains voxel size and volume size parameters.
    radius : float
        Radius (as a percentage of the image diagonal) used for the mask.
        Default is 0.99, which masks close to the image boundaries.

    Returns:
    --------
    masked_recon : numpy.ndarray
        The volume after applying a soft circular mask.
    """
    nz, ny, nx = recon.shape

    # Extract voxel size from TIGRE geometry
    voxel_size_y = geo.dVoxel[1]
    voxel_size_x = geo.dVoxel[2]

    # Extract total image size (in mm) from TIGRE geometry
    size_y = geo.sVoxel[1]
    size_x = geo.sVoxel[2]

    # Generate a meshgrid for 2D slice (Y, X coordinates)
    x_range = (nx - 1) / 2
    y_range = (ny - 1) / 2
    Y, X = np.ogrid[-y_range:y_range+1, -x_range:x_range+1]

    # Compute Euclidean distance of each pixel from the center (in mm)
    dist_from_center = np.sqrt((X * voxel_size_x) ** 2 + (Y * voxel_size_y) ** 2)

    # Radius in mm (relative to image size) 
    radius_mm = radius * max(size_x, size_y) / 2

    # Define ramp for smooth masking (sinusoidal taper)
    r = ((voxel_size_x * voxel_size_y) / np.pi) ** 0.5

    mask = (radius_mm - dist_from_center).clip(-r, r)
    mask *= (0.5 * np.pi) / r
    np.sin(mask, out=mask)
    mask = 0.5 + 0.5 * mask  # 0–1 normalize

    # Apply 2D mask to each slice of the 3D volume
    # recon = recon * mask[np.newaxis, :, :]

    # Allocate output array
    recon_masked = np.empty_like(recon)
    #  Apply 2D mask parallel masking
    def apply_mask_to_slice(i):
        #recon[i] *= mask  # in-place masking
        recon_masked[i] = recon[i] * mask
    n_jobs = multiprocessing.cpu_count()
    Parallel(n_jobs=n_jobs, backend="threading", prefer="threads")(
        delayed(apply_mask_to_slice)(i) for i in range(nz)
    )

    return recon_masked

#
#%% Apply masking in parallel 

imgFDK_masked = apply_circular_mask(imgFDK, geo, radius=0.99) # you can chose specific radius size

#%% --- Visualize ---
#% 
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.imshow(imgFDK[0], cmap='gray')
plt.title("FDK")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(imgFDK_masked [0], cmap='gray')
plt.title("Masked with circular ROI")
plt.axis("off")
plt.show()


# %%