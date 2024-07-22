import numpy as np
import matplotlib.pyplot as plt

PATH = "D:/calibration_paper_data/"
DEVICE = "invision"  # "invision", "tropus", "svot"
PHANTOM = "13"  # note that 2, 4, 6, 8 have the names 2.2, 2.3 etc
WAVELENGTH = "800"  # "750", "800", "850", "900"

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, layout="constrained")
selector = np.s_[:, :]
if DEVICE == "svot":
    selector = np.s_[:, :, 180]

ax1.imshow(np.load(f"{PATH}/{DEVICE}/recon/P.5.{PHANTOM}_{WAVELENGTH}.npy")[selector], cmap="gray")
ax2.imshow(np.load(f"{PATH}/{DEVICE}/recon/P.5.{PHANTOM}_{WAVELENGTH}.npy")[selector], cmap="gray")

if DEVICE == "svot":
    ax2.imshow(np.load(f"{PATH}/{DEVICE}/labels/P.5.{PHANTOM}.npz")["data"][selector], alpha=0.5)
    ax2.contour(np.load(f"{PATH}/{DEVICE}/labels/P.5.{PHANTOM}.npz")["data"][selector])
    ax3.imshow(np.load(f"{PATH}/{DEVICE}/mua/P.5.{PHANTOM}_{WAVELENGTH}.npz")["data"][selector])
    ax4.imshow(np.load(f"{PATH}/{DEVICE}/musp/P.5.{PHANTOM}_{WAVELENGTH}.npz")["data"][selector])
else:
    ax2.imshow(np.load(f"{PATH}/{DEVICE}/labels/P.5.{PHANTOM}.npy"), alpha=0.5)
    ax2.contour(np.load(f"{PATH}/{DEVICE}/labels/P.5.{PHANTOM}.npy"))
    ax3.imshow(np.load(f"{PATH}/{DEVICE}/mua/P.5.{PHANTOM}_{WAVELENGTH}.npy"))
    ax4.imshow(np.load(f"{PATH}/{DEVICE}/musp/P.5.{PHANTOM}_{WAVELENGTH}.npy"))

plt.show()
plt.close()
