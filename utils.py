from typing import List, Dict
import numpy as np

def rmse(prediction, target, mask=None):
    if mask is None:
        return np.sqrt(((prediction - target) ** 2).mean())
    else:
        if np.amax(mask)==255:
            mask/=255.0
        return np.sqrt(((prediction[mask==1] - target[mask==1]) ** 2).mean())

def cartesian2pixel(x, y, img_size):
    row = int(y  * (img_size/2))
    col = int(x  * (img_size/2))
    return row, col

def pixel2cartesian(row, col, img_size):
    y = -(row - (img_size/2)) / (img_size/2)
    x =  (col - (img_size/2)) / (img_size/2)
    return x, y

def pose2params(vec : np.array, pixel_domain=False) -> List:
    t_x, t_y = vec[0], vec[1]
    s_cos, s_sin = vec[2], vec[3]
    scale = np.sqrt(s_cos**2 + s_sin**2)
    cos, sin = s_cos/scale, s_sin/scale
    theta = np.rad2deg(np.arctan2(sin, cos))

    if pixel_domain:
        t_x, t_y = cartesian2pixel(t_x, t_y, 224)
        return t_y, -t_x, theta, scale
    return t_x, t_y, theta, scale

def get_appearance_lens(sets : Dict, img_names : List) -> List:
    sets = sets[sets['name'].isin(img_names)][['eyes','nose','mouth','jaw','hair']].to_numpy()
    lens = []
    for idx in range(5):
        lens.append(sets[0,idx].shape[0])
    return lens

def get_angle_diff(b1, b2):
    dist = (b2-b1)%360.0
    if dist>=180.0:
        dist-=360
    return np.abs(dist)
