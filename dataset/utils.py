
import numpy as np
import copy
import cv2
import os

def load_parts() -> dict:
    # part characteristics.
    sexes = ['male', 'female']
    races = ['black', 'white', 'asian']
    parts = ['eyes', 'hair', 'jaw', 'mouth', 'nose']

    # load part imgs and convert them to arrays.
    part_arrays = {}
    for part in parts:
        part_arrays[part] = []
    
    for sex in sexes:
        for race in races:
            for part in parts:
                for idx in range(7):
                    # Load part and its individual mask.
                    part_arr = cv2.imread(os.path.join('./dataset/face_parts', sex, race, part, part+'_'+str(idx)+'.png'), -1)
                    part_arrays[part].append(resize_part(part_arr, part)/255.0)

    return part_arrays

def scale_part(part_img, scale):
    # Center image.
    T_trans_neg = np.array([[1, 0, -part_img.shape[1]/2],[0, 1, -part_img.shape[0]/2],[0, 0, 1]])
    # Scale image.
    T_scale = np.array([[scale, 0, 0],[0, scale, 0],[0, 0, 1]])
    # Un-center image.
    T_trans_plus =  np.array([[1, 0, part_img.shape[1]/2],[0, 1, part_img.shape[0]/2],[0, 0, 1]])

    # Calculate total transform.
    T = T_trans_plus @ T_scale @ T_scale @ T_trans_neg
    T = np.float32(T.flatten()[:6].reshape(2,3))

    # Apply transform.
    return cv2.warpAffine(part_img, T, (part_img.shape[1], part_img.shape[0]))

def translate_part(part_img, x):
    # Make transform matrix.
    T_trans = np.array([[1, 0, 0],[0, 1, x],[0, 0, 1]])
    T = np.float32(T_trans.flatten()[:6].reshape(2,3))
    
    # Apply transform.
    return cv2.warpAffine(part_img, T, (part_img.shape[1], part_img.shape[0]))

def get_hair_mask(shape):
    # Ellipsis config.
    center_coordinates = (320,240)
    axesLength = (175, 210)
    angle, startAngle, endAngle = 0, 0, 360
    thickness = -1
    color = (255, 0, 0)

    # Create ellipsis.
    ellipsis = np.zeros(shape)
    cv2.ellipse(ellipsis, center_coordinates, axesLength, angle, 
        startAngle, endAngle, color, thickness)
    ellipsis[385:,:] = np.zeros(ellipsis[385:,:].shape)

    # Create block 
    rect = np.zeros((640,640))
    rect[245:385, 150:490] = np.ones(rect[245:385, 150:490].shape)*255

    ellipsis+=rect

    masks = []
    for p in ['eyes', 'nose', 'mouth']:
        mask = cv2.imread(f'./dataset/masks/{p}_mask_new.png',  0)
        if p=='mouth': 
            mask=mask/255.0
        masks.append(mask)

    # Zero-out mask area.
    for idx, mask in enumerate(masks):
        masks[idx] = np.ones(mask.shape) - mask

    # Apply mask.
    for mask in masks:
        ellipsis = np.multiply(ellipsis, mask)
    ellipsis[ellipsis>255] = 255
    return ellipsis 

def get_jaw_mask(shape):
    # Ellipsis config.
    center_coordinates = (320,320)
    axesLength = (152, 210)
    angle, startAngle, endAngle = 0, 0, 360
    thickness = -1
    color = (255, 0, 0)

    # Create ellipsis.
    ellipsis = np.zeros(shape)
    cv2.ellipse(ellipsis, center_coordinates, axesLength, angle, 
        startAngle, endAngle, color, thickness)
    ellipsis[0:385,:] = np.zeros(ellipsis[0:385,:].shape)

    masks = []
    for p in ['eyes', 'nose', 'mouth']:
        mask = cv2.imread(f'./dataset/masks/{p}_mask_new.png',  0)
        if p=='mouth': 
            mask=mask/255.0
        masks.append(mask)

    # Zero-out mask area.
    for idx, mask in enumerate(masks):
        masks[idx] = np.ones(mask.shape) - mask

    # Apply mask.
    for mask in masks:
        ellipsis = np.multiply(ellipsis, mask)

    return ellipsis

def get_face_mask(resize=False):
    canvas_shape = (640, 640)

    hair_mask = get_hair_mask(canvas_shape)/255.0
    jaw_mask = get_jaw_mask(canvas_shape)/255.0

    masks = []
    for p in ['eyes', 'nose', 'mouth']:
        mask = cv2.imread(f'./dataset/masks/{p}_mask_new.png',  0)
        if p=='mouth': 
            mask=mask/255.0
        masks.append(mask)

    eyes_mask  = masks[0]
    nose_mask  = masks[1]
    mouth_mask = masks[2]

    face_mask = eyes_mask+ nose_mask+ mouth_mask +jaw_mask+ hair_mask
    face_mask[face_mask>1]=1

    if resize:
        return cv2.resize(face_mask, (224,224))

    return face_mask

def crop_hair(part_img):
    # Crop hair and eyes, nose areas.
    hair_mask = get_hair_mask(part_img.shape)
    # Mask hair.
    part_img=np.multiply(part_img, hair_mask/255.0)

    return part_img

def crop_jaw(part_img):
    # mask part.
    jaw_mask = get_jaw_mask(part_img.shape)
    part_img=np.multiply(part_img, jaw_mask/255.0)

    return part_img

def resize_part(part_img, part_name):
    mask = part_img[:,:,-1]/255.0
    part_img = np.multiply(part_img[:,:,0], mask)

    # If the part is nose, mouth or eyes, crop the rectangular background.
    if part_name == 'nose' or part_name == 'mouth' or part_name == 'eyes': 
        return part_img[42:125, 120:359].reshape((-1))
    else:
        canvas = np.multiply(np.ones((640,640))*100, np.ones((640,640))-mask)
        if part_name == 'hair':
            part_arr = crop_hair(part_img+canvas)
        else:
            part_arr = crop_jaw(part_img+canvas)
        return reshape_part(part_arr, part_name)

def reshape_part(part_img, part_name):
    if part_name == 'hair':
        mask = get_hair_mask(part_img.shape)
    elif part_name == 'jaw':
        mask = get_jaw_mask(part_img.shape)
    
    part_pixels = []
    for r in range(640):
        for c in range(640):
            if mask[r,c]==255.0:
                part_pixels.append(part_img[r,c])

    return np.array(part_pixels)

def reshape_components(projected, part_name):
    shape=(640,640)
    if part_name == 'hair':
        mask = get_hair_mask(shape)
    elif part_name == 'jaw':
        mask = get_jaw_mask(shape)

    pxl_idx = 0
    canvas = np.zeros(shape)
    for r in range(640):
        for c in range(640):
            if mask[r,c]==255.0:
                canvas[r,c]=projected[pxl_idx]
                pxl_idx+=1
    return canvas

def isolate_pcs(output, part_idx):
    components = {}
    num_components = [0, 25+6, 11+6, 12+6, 15+6, 28+6]
    indices = [sum(num_components[:i+1]) for i in range(6)]
    for idx, part_name in enumerate(['eyes', 'nose', 'mouth', 'jaw', 'hair']):
        part_appearance = copy.deepcopy(output[:, indices[idx]:indices[idx+1]][:,6:])
        # Save 0-mean, 1-var appearance components.
        components[part_name] = part_appearance[part_idx]
    return components

def mix_components(real : dict, pred : dict, pc : int, part_name : str) -> dict:
    real_mod = copy.deepcopy(real)
    if type(part_name) is list:
        for part in part_name:
            real_mod[part][pc] = pred[part][pc]
    else:
        real_mod[part_name][pc] =real_mod[part_name][pc]

    return real_mod