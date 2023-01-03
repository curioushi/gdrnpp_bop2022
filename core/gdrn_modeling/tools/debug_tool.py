import cv2
import torch
import numpy as np

def dump_image(name, x, prefix='/home/shihaoqi/tmp/', normalize=True):
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    if not (2 <= x.ndim <= 3):
        print(f'image dimension wrong: {x.shape}')
        return
    if x.ndim == 3:
        if x.shape[0] == 1 or x.shape[0] == 3:
            x = x.transpose([1, 2, 0])
        if not (x.shape[-1] == 1 or x.shape[-1] == 3):
            print(f'image shape wrong: {x.shape}')
        if x.shape[-1] == 1:
            x = x[:, :, 0]
    if normalize and (x.dtype == np.float32 or x.dtype == np.float64):
        xmin = x.min()
        xmax = x.max()
        x = 255 * ((x - xmin) / (xmax - xmin))
    elif x.dtype == np.bool:
        x = 255 * x
    x = x.astype(np.uint8)
    cv2.imwrite(prefix + name + '.png', x)
