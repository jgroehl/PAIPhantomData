import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Create a sample 3D binary mask
binary_mask = np.random.randint(0, 2, (100, 100, 100))

# Extract the surface mesh
verts, faces, _, _ = measure.marching_cubes(binary_mask, level=0)

# Plot the surface mesh
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
mesh = Poly3DCollection(verts[faces], alpha=0.1)
face_color = [0.5, 0.5, 0.5]  # Alternative: use gray color
mesh.set_facecolor(face_color)
ax.add_collection3d(mesh)

ax.set_xlim(0, binary_mask.shape[0])
ax.set_ylim(0, binary_mask.shape[1])
ax.set_zlim(0, binary_mask.shape[2])

plt.show()