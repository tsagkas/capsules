from dataset.utils import scale_part, translate_part, crop_hair, crop_jaw, get_face_mask
import matplotlib.pyplot as plt
from models.pca import pc2img
import numpy as np
import cv2
import os

def generate_face(face_args=None, components=None, show=False) -> np.array:
    part_names = ['jaw', 'hair', 'nose', 'eyes', 'mouth']
    parts = []

    # Synthetic face generation from parameters.
    if face_args is not None:
        # Get parts.
        sex, race, indices = face_args
        for idx, part_name in enumerate(part_names):
            part_img, part_mask = get_part(part_name, sex, race, index=indices[idx])
            parts.append(mask_part(part_img, part_name, mask=part_mask))

    # Synthetic face generation from appearance vector.
    if components is not None:
        # Transform the 5 components to image. 
        for part_name in part_names:
            part_img = pc2img(components[part_name], part_name)
            parts.append(mask_part(part_img, part_name)*255.0)

    # Create face from parts.
    face = np.multiply(arrange_parts(parts), get_face_mask())
    canvas = np.ones((640,640))*100-get_face_mask()*100 + face
    canvas = cv2.resize(canvas, (224,224))

    # Plot synthetic face.
    if show:
        plt.imshow(canvas, cmap='gray')
        plt.show()

    return canvas

def get_part(part_name : str, sex : str, race : str, index : int, scale=1.05) -> np.array:
    # Load part image.
    part_img = cv2.imread(os.path.join('./dataset/face_parts', sex, race, part_name, part_name+'_'+str(index)+'.png'), -1)
    mask = part_img[:,:,-1]/255.0
    part_img = np.multiply(part_img[:,:,0], mask)

    return part_img, mask

def mask_part(part_img : np.array, part_name : str, mask=None):
    if part_name == 'hair':
        if mask is not None:
            # Only mask the hair if we generate face from args.
            canvas = np.multiply(np.ones((640,640))*100, np.ones((640,640))-mask)
            part_img = crop_hair(part_img+canvas)

    elif part_name == 'jaw':
        if mask is not None:
            # Only mask the jaw if we generate face from args.
            canvas = np.multiply(np.ones((640,640))*100, np.ones((640,640))-mask)
            part_img = crop_jaw(part_img+canvas)
    else:
        part_img = crop_enm(scale_part(part_img, scale=1.05), part=part_name)

    return part_img

def crop_enm(part_img : np.array, part : str) -> np.array:
    # Create empty canvas.
    IMG_DIMS = 640 
    canvas = np.zeros((IMG_DIMS, IMG_DIMS))

    # How much to translate downwards the part.
    translate_x = {
        'eyes' : 186,
        'nose' : 180+80,
        'mouth': 190+80+70,
    }

    # Position the part on the canvas.
    y_offset = int((IMG_DIMS-part_img.shape[1])/2)
    canvas[0:part_img.shape[0], y_offset:part_img.shape[1]+y_offset] = part_img
    canvas_translated = translate_part(canvas, translate_x[part])

    # Load the mask.
    mask  = cv2.imread(f'./dataset/masks/{part}_mask_new.png', 0)
    if part == 'mouth': 
        mask=mask/255.0

    # Mask the canvas.
    return np.multiply(canvas_translated, mask)

def arrange_parts(parts : list) -> np.array:
    # Connect hair and jaw parts.
    face = parts[0] + parts[1]

    # Load masks of parts that go on top and apply them
    # to the hair and jaw parts.
    masks = []
    for p in ['eyes', 'nose', 'mouth']:
        mask = cv2.imread(f'./dataset/masks/{p}_mask_new.png',  0)
        if p=='mouth': 
            mask=mask/255.0
        masks.append(mask)

    # Zero-out mask area.
    for idx, mask in enumerate(masks):
        masks[idx] = np.ones(mask.shape) - mask

    for i, j in zip([1,0,2], [2,3,4]):
        face = np.multiply(face, masks[i])
        face = face + parts[j]
    return face