import json
import os
import simpa as sp
import numpy as np

DEVICES = ["tropus", "invision", "svot"]
path = r"D:\calibration_paper_data/material_mapping.json"

with open(path, "r+") as jsonfile:
    mapping = json.load(jsonfile)

for phantom_key in mapping.keys():
    print(phantom_key)
    for device in DEVICES:
        print(f"\t{device}")
        if device == "svot":
            segmentation = np.load(f"D:\calibration_paper_data/{device}/labels/{phantom_key}.npz")["data"]
            wavelengths = [700, 740, 800, 840, 900]
        else:
            segmentation = np.load(f"D:\calibration_paper_data/{device}/labels/{phantom_key}.npy")
            wavelengths = [700, 750, 800, 850, 900]

        if np.min(segmentation) > 0:
            raise ValueError("The segmentation mask had an invalid definition and no water background was defined!")

        for wavelength in wavelengths:
            mua = np.zeros_like(segmentation).astype(float)
            musp = np.zeros_like(segmentation).astype(float)
            # assign phantom materials
            for material_key in mapping[phantom_key]:
                if int(material_key) > 0:
                    material_name = mapping[phantom_key][material_key]
                    material = np.load(fr"D:\calibration_paper_data\DIS\processed/{material_name}.a/{material_name}.npz")
                    mua[segmentation==int(material_key)] = material["mua"][material["wavelengths"] == wavelength]
                    musp[segmentation == int(material_key)] = material["mus"][material["wavelengths"] == wavelength] * 0.3
            # assign water
            mua[segmentation == 0] = sp.MOLECULE_LIBRARY.water().spectrum.get_value_for_wavelength(wavelength)
            musp[segmentation == 0] = sp.MOLECULE_LIBRARY.water().scattering_spectrum.get_value_for_wavelength(wavelength)

            if device == "svot":
                np.savez_compressed(f"D:\calibration_paper_data/{device}/mua/{phantom_key}_{wavelength}.npz",
                                    data=mua)
                np.savez_compressed(f"D:\calibration_paper_data/{device}/musp/{phantom_key}_{wavelength}.npz",
                                    data=musp)
            else:
                np.save(f"D:\calibration_paper_data/{device}/mua/{phantom_key}_{wavelength}.npy", mua)
                np.save(f"D:\calibration_paper_data/{device}/musp/{phantom_key}_{wavelength}.npy", musp)


