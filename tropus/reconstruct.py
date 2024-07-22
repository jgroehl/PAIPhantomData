import patato as pat
import numpy as np
from patato.io.ipasc.read_ipasc import IPASCInterface
import glob
import time

data_source = "testing"

def reconstruct(path, sound_speed=1477):
    pa_data = pat.PAData(IPASCInterface(path))
    time_factor = 1
    detector_factor = 1
    preproc = pat.DefaultMSOTPreProcessor(time_factor=time_factor, detector_factor=detector_factor,
                                          hilbert=True, lp_filter=None, hp_filter=None,
                                          irf=False)
    patato_recon = pat.ReferenceBackprojection(field_of_view=(0.032, 0.032, 0.032), n_pixels=(300, 1, 300))
    new_t1, d1, _ = preproc.run(pa_data.get_time_series(), pa_data)
    recon, _, _ = patato_recon.run(new_t1, pa_data, sound_speed, **d1)
    return (np.asarray(np.squeeze(recon.raw_data)))[6:-6, 6:-6].copy()


for file in glob.glob(r"D:\calibration_paper_data\tropus\raw/*.hdf5"):
    print(file)
    save_file_path = file.replace("raw", "recon").replace("_ipasc.hdf5", ".npy")
    print("===================", "\nsaving\n\t", file, "\nto\n\t", save_file_path)
    recon = reconstruct(file).T
    np.save(save_file_path, recon)
