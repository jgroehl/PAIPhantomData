import nrrd
import numpy as np

from scipy.ndimage import gaussian_filter, binary_erosion


def linearly_interpolate_slices(data, segmented_slices, label=3, axis=2, base=False):
    slice_indices = np.array(segmented_slices)
    interpolated_data = np.zeros_like(data)

    # Iterate over each z pair of segmented slices
    for k in range(len(slice_indices) - 1):
        z0, z1 = slice_indices[k], slice_indices[k + 1]

        if not base:
            if axis == 2:
                mask_z0 = (data[:, :, z0] == label).astype(float)
                mask_z1 = (data[:, :, z1] == label).astype(float)
            elif axis == 1:
                mask_z0 = (data[:, z0, :] == label).astype(float)
                mask_z1 = (data[:, z1, :] == label).astype(float)
            elif axis == 0:
                mask_z0 = (data[z0, :, :] == label).astype(float)
                mask_z1 = (data[z1, :, :] == label).astype(float)
        else:
            if axis == 2:
                mask_z0 = (data[:, :, z0] >= 2).astype(float)
                mask_z1 = (data[:, :, z1] >= 2).astype(float)
            elif axis == 1:
                mask_z0 = (data[:, z0, :] >= 2).astype(float)
                mask_z1 = (data[:, z1, :] >= 2).astype(float)
            elif axis == 0:
                mask_z0 = (data[z0, :, :] >= 2).astype(float)
                mask_z1 = (data[z1, :, :] >= 2).astype(float)

        # Interpolate all intermediate slices
        for zi in range(z0, z1):
            t = (zi - z0) / (z1 - z0)  # Normalized position between z0 and z1
            interpolated_mask = mask_z0 * (1 - t) + mask_z1 * t
            if axis == 2:
                interpolated_data[:, :, zi] = (interpolated_mask > 0.5).astype(int)
            elif axis == 1:
                interpolated_data[:, zi, :] = (interpolated_mask > 0.5).astype(int)
            elif axis == 0:
                interpolated_data[zi, :, :] = (interpolated_mask > 0.5).astype(int)

    return interpolated_data


def smooth_along_axes(data, sigma=None):
    if sigma is None:
        sigma = (10, 10, 30)
    smooth_data = data.copy().astype(float)
    smooth_data = gaussian_filter(smooth_data, sigma=sigma, mode=["constant", "constant", "reflect"])
    smooth_data = (np.abs(smooth_data) > 0.5).astype(int)
    return smooth_data


def interpolate(data, label=3, axis=2, base=False):

    print("Axis:", axis)

    # First extract the fully-segmented slices
    if axis == 0:
        segmented_slices = np.argwhere(np.sum(binary_erosion(data == label, iterations=2), axis=(1, 2)) > 0)
    elif axis == 1:
        segmented_slices = np.argwhere(np.sum(binary_erosion(data == label, iterations=2), axis=(0, 2)) > 0)
    elif axis == 2:
        segmented_slices = np.argwhere(np.sum(binary_erosion(data == label, iterations=2), axis=(0, 1)) > 0)

    segmented_slices = [a[0] for a in segmented_slices]
    if axis == 2:
        segmented_slices.insert(0, 0)
        segmented_slices.append(359)
    print("\tManually segmented slices:", segmented_slices)

    print("\tInterpolating...")
    interpolated_data = linearly_interpolate_slices(data, segmented_slices, label=label, axis=axis, base=base)
    return interpolated_data


def interpolate_z(path, sigma_inc, sigma_base):
    data, metadata = nrrd.read(path)

    interpolated_data_z_inc_1 = interpolate(data, label=3, axis=2)
    interpolated_data_inc_1 = interpolated_data_z_inc_1

    interpolated_data_z_inc_2 = interpolate(data, label=4, axis=2)
    interpolated_data_inc_2 = interpolated_data_z_inc_2

    interpolated_data_z_base = interpolate(data, base=True, label=2, axis=2)
    interpolated_data_base = interpolated_data_z_base

    print("Smoothing Inclusion...")
    smoothed_data_inc_1 = smooth_along_axes(interpolated_data_inc_1, sigma_inc)
    smoothed_data_inc_2 = smooth_along_axes(interpolated_data_inc_2, sigma_inc)

    print("Smoothing Base...")
    smoothed_data_base = smooth_along_axes(interpolated_data_base, sigma_base)

    print("Creating Final Segmentation")
    final_segmentation = np.ones_like(smoothed_data_base)

    print("Assigning base material segmentation")
    final_segmentation[smoothed_data_base == 1] = 2

    if np.isin(3, data):
        print("Assigning inclusion segmentation 1")
        final_segmentation[smoothed_data_inc_1 == 1] = 3

    if np.isin(4, data):
        print("Assigning inclusion segmentation 2")
        final_segmentation[smoothed_data_inc_2 == 1] = 4

    nrrd.write(path.replace("nrrd_manual_labels", "nrrd_extended_labels"), final_segmentation, metadata)


def interpolate_xyz(path, sigma_inc, sigma_base):
    data, metadata = nrrd.read(path)

    interpolated_data_z_inc_1 = interpolate(data, label=3, axis=2)
    interpolated_data_y_inc_1 = interpolate(data, label=3, axis=1)
    interpolated_data_x_inc_1 = interpolate(data, label=3, axis=0)
    interpolated_data_inc_1 = (interpolated_data_z_inc_1 + interpolated_data_y_inc_1 + interpolated_data_x_inc_1) / 3
    interpolated_data_z_inc_2 = interpolate(data, label=4, axis=2)
    interpolated_data_y_inc_2 = interpolate(data, label=4, axis=1)
    interpolated_data_x_inc_2 = interpolate(data, label=4, axis=0)
    interpolated_data_inc_2 = (interpolated_data_z_inc_2 + interpolated_data_y_inc_2 + interpolated_data_x_inc_2) / 3

    # We can always assume that the base is roughly symmetrical along the z direction
    interpolated_data_base = interpolate(data, base=True, label=2, axis=2)

    print("Smoothing Inclusion...")
    smoothed_data_inc_1 = smooth_along_axes(interpolated_data_inc_1, sigma=sigma_inc)
    smoothed_data_inc_2 = smooth_along_axes(interpolated_data_inc_2, sigma=sigma_inc)

    print("Smoothing Base...")
    smoothed_data_base = smooth_along_axes(interpolated_data_base, sigma=sigma_base)

    print("Creating Final Segmentation")
    final_segmentation = np.ones_like(smoothed_data_base)

    print("Assigning base material segmentation")
    final_segmentation[smoothed_data_base == 1] = 2

    if np.isin(3, data):
        print("Assigning inclusion segmentation 1")
        final_segmentation[smoothed_data_inc_1 == 1] = 3

    if np.isin(4, data):
        print("Assigning inclusion segmentation 2")
        final_segmentation[smoothed_data_inc_2 == 1] = 4

    nrrd.write(path.replace("nrrd_manual_labels", "nrrd_extended_labels"), final_segmentation, metadata)


interpolate_z(r"D:\calibration_paper_data\nrrd_manual_labels\P.5.1_700-labels.nrrd",
             sigma_inc=(10, 10, 30), sigma_base=(10, 10, 30))

interpolate_xyz(r"D:\calibration_paper_data\nrrd_manual_labels\P.5.2.2_700-labels.nrrd",
              sigma_inc=(8, 8, 8), sigma_base=(10, 10, 30))

interpolate_z(r"D:\calibration_paper_data\nrrd_manual_labels\P.5.2.3_700-labels.nrrd",
              sigma_inc=(5, 5, 30), sigma_base=(10, 10, 30))

interpolate_z(r"D:\calibration_paper_data\nrrd_manual_labels\P.5.3_700-labels.nrrd",
              sigma_inc=(10, 10, 30), sigma_base=(10, 10, 30))

interpolate_z(r"D:\calibration_paper_data\nrrd_manual_labels\P.5.4.2_700-labels.nrrd",
             sigma_inc=(3, 3, 30), sigma_base=(3, 3, 30))

interpolate_z(r"D:\calibration_paper_data\nrrd_manual_labels\P.5.4.3_700-labels.nrrd",
              sigma_inc=(5, 5, 30), sigma_base=(10, 10, 30))

interpolate_z(r"D:\calibration_paper_data\nrrd_manual_labels\P.5.5_700-labels.nrrd",
              sigma_inc=(10, 10, 30), sigma_base=(10, 10, 30))

interpolate_z(r"D:\calibration_paper_data\nrrd_manual_labels\P.5.6.2_700-labels.nrrd",
              sigma_inc=(8, 8, 30), sigma_base=(10, 10, 30))

interpolate_z(r"D:\calibration_paper_data\nrrd_manual_labels\P.5.6.3_700-labels.nrrd",
              sigma_inc=(5, 5, 30), sigma_base=(10, 10, 30))

interpolate_z(r"D:\calibration_paper_data\nrrd_manual_labels\P.5.7_700-labels.nrrd",
              sigma_inc=(10, 10, 30), sigma_base=(10, 10, 30))

interpolate_z(r"D:\calibration_paper_data\nrrd_manual_labels\P.5.9_700-labels.nrrd",
              sigma_inc=(10, 10, 30), sigma_base=(10, 10, 30))

interpolate_z(r"D:\calibration_paper_data\nrrd_manual_labels\P.5.11_700-labels.nrrd",
              sigma_inc=(10, 10, 30), sigma_base=(10, 10, 30))

interpolate_xyz(r"D:\calibration_paper_data\nrrd_manual_labels\P.5.8.2_700-labels.nrrd",
                sigma_inc=(8, 8, 8), sigma_base=(10, 10, 30))

interpolate_xyz(r"D:\calibration_paper_data\nrrd_manual_labels\P.5.10.2_700-labels.nrrd",
                sigma_inc=(8, 8, 8), sigma_base=(10, 10, 30))

interpolate_z(r"D:\calibration_paper_data\nrrd_manual_labels\P.5.10.3_700-labels.nrrd",
              sigma_inc=(10, 10, 30), sigma_base=(10, 10, 30))

interpolate_z(r"D:\calibration_paper_data\nrrd_manual_labels\P.5.13_700-labels.nrrd",
              sigma_inc=(10, 10, 60), sigma_base=(10, 10, 60))

interpolate_z(r"D:\calibration_paper_data\nrrd_manual_labels\P.5.14_700-labels.nrrd",
              sigma_inc=(5, 5, 30), sigma_base=(10, 10, 30))

interpolate_z(r"D:\calibration_paper_data\nrrd_manual_labels\P.5.15_700-labels.nrrd",
              sigma_inc=(10, 10, 30), sigma_base=(10, 10, 30))

interpolate_z(r"D:\calibration_paper_data\nrrd_manual_labels\P.5.16_700-labels.nrrd",
               sigma_inc=(5, 5, 30), sigma_base=(10, 10, 30))

interpolate_z(r"D:\calibration_paper_data\nrrd_manual_labels\P.5.19_700-labels.nrrd",
               sigma_inc=(10, 10, 30), sigma_base=(10, 10, 30))

interpolate_xyz(r"D:\calibration_paper_data\nrrd_manual_labels\P.5.20_700-labels.nrrd",
                sigma_inc=(8, 8, 8), sigma_base=(10, 10, 30))

interpolate_z(r"D:\calibration_paper_data\nrrd_manual_labels\P.5.21_700-labels.nrrd",
               sigma_inc=(10, 10, 30), sigma_base=(10, 10, 30))

interpolate_z(r"D:\calibration_paper_data\nrrd_manual_labels\P.5.22_700-labels.nrrd",
               sigma_inc=(5, 5, 10), sigma_base=(10, 10, 30))

interpolate_z(r"D:\calibration_paper_data\nrrd_manual_labels\P.5.23_700-labels.nrrd",
               sigma_inc=(10, 10, 30), sigma_base=(10, 10, 30))

interpolate_z(r"D:\calibration_paper_data\nrrd_manual_labels\P.5.24_700-labels.nrrd",
                sigma_inc=(10, 10, 15), sigma_base=(10, 10, 30))

interpolate_z(r"D:\calibration_paper_data\nrrd_manual_labels\P.5.27_700-labels.nrrd",
                sigma_inc=(10, 10, 30), sigma_base=(10, 10, 30))

interpolate_z(r"D:\calibration_paper_data\nrrd_manual_labels\P.5.29_700-labels.nrrd",
                sigma_inc=(10, 10, 30), sigma_base=(10, 10, 30))

interpolate_z(r"D:\calibration_paper_data\nrrd_manual_labels\P.5.31_700-labels.nrrd",
                sigma_inc=(10, 10, 30), sigma_base=(10, 10, 30))

interpolate_z(r"D:\calibration_paper_data\nrrd_manual_labels\P.5.32_700-labels.nrrd",
                sigma_inc=(10, 10, 30), sigma_base=(10, 10, 30))