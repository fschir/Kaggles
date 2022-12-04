import numpy as np
import pydicom
import matplotlib.pyplot as plt
import os
import glob
from pathlib import Path

import torch
import torchvision.transforms as T
from torch import tensor, Tensor

def rescale_img_to_hu(dcm_ds):
    """Rescales the image to Hounsfield unit."""
    data = dcm_ds.pixel_array
    if dcm_ds.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data
    data = np.array(data * dcm_ds.RescaleSlope + dcm_ds.RescaleIntercept)
    data = (data - data.min()) / (data.max() - data.min())
    return data

def normalize(img: np.ndarray):
    # https://www.kaggle.com/code/nguynththanhho/rsna-breast-cancer-preprocescing
    m = int(np.max(img))
    hist = np.histogram(img, bins=m + 1, range=(0, m + 1))
    hist = hist[0] / img.size
    scale = (255 * np.cumsum(hist))
    return np.array([scale[i] for i in img.ravel()]).reshape(img.shape)

def show_images_for_patient(patient_id, is_local=False):
    local_dir = Path('data/rsna/train_images')
    remote_dir = Path('/kaggle/input/rsna-breast-cancer-detection/')
    patient_dir = (local_dir / str(patient_id)) if is_local else (remote_dir / str(patient_id))
    num_images = len(glob.glob(f"{patient_dir}/*"))
    print(f"Number of images for patient: {num_images}")
    fig, axs = plt.subplots(2, 2, figsize=(24,15))
    axs = axs.flatten()
    for i, img_path in enumerate(list(Path(patient_dir).iterdir())):
        ds = pydicom.dcmread(img_path)
        axs[i].imshow(rescale_img_to_hu(ds), cmap="bone")