from dataset.utils import load_parts, resize_part, reshape_components
from sklearn.decomposition import PCA
import pickle as pk
import numpy as np
import cv2
import os

def train_PCA(percentage=0.95):
    print('=> Loading parts..')
    parts_dict = load_parts()

    # Train a PCA Model for each part type.
    for part_name in ['jaw', 'hair', 'nose', 'eyes', 'mouth']:
        print(f'=> Training PCA model for: {part_name}')
        # get the part images of the part-type.
        parts = parts_dict[part_name]

        # Train PCA.
        pca = PCA(percentage)
        pca.fit(parts)
        
        # Save PCA model
        if not os.path.exists('models/pca'):
            os.makedirs('models/pca')
        pk.dump(pca, open(f'models/pca/pca_{part_name}.pkl','wb'))

    print('=> Done!')

def img2pc(part : np.array, part_name : str) -> np.array:
    # Load the PCA model.
    pca = pk.load(open(f'models/pca/pca_{part_name}.pkl','rb'))
    # Reshape img.
    part = part.reshape((1,-1))
    # get PCs with unit-variance.
    components = pca.transform(part)/np.sqrt(pca.explained_variance_)

    return components

def pc2img(components : np.array, part_name : str) -> np.array:
    # Load PCA model.
    pca = pk.load(open(os.path.join('models/pca', f'pca_{part_name}.pkl'),'rb'))
    # PC space => Image space (multiply with sqrt(eigenvalue)).
    projected = pca.inverse_transform(components*np.sqrt(pca.explained_variance_))

    if part_name=='hair' or part_name=='jaw':
        return reshape_components(projected, part_name)
    else:
        img_shape = (83, 239)
        # Reshape img.
        part_img = projected.reshape(img_shape) 

        # Create empty canvas.
        canvas = np.zeros((168, 480))
        canvas[42:125, 120:359] = part_img  

        mask = np.load('./dataset/masks/enm_mask.npy')
        canvas = np.multiply(canvas, mask)

        return canvas

def get_part_components() -> dict:
    part_names = ['eyes', 'jaw', 'hair', 'nose', 'mouth']
    races = ['asian', 'black', 'white']
    sexes = ['female', 'male']

    # Dictionary that holds the PCs for each part image.
    part_components = {}

    # Go through all part-images and map them to pcs.
    for part_name in part_names:
        part_components[part_name] = {}
        for sex in sexes:
            part_components[part_name][sex] = {}
            for race in races:
                part_components[part_name][sex][race] = {}
                for i in range(7):
                    # load part-image
                    part_img = cv2.imread(os.path.join('./dataset/face_parts', sex, race, part_name, part_name+'_'+str(i)+'.png'), -1)
                    part_img = resize_part(part_img, part_name)/255.0
                    
                    # Get components and save them.
                    components = img2pc(part_img, part_name)
                    part_components[part_name][sex][race][f'{part_name}_{str(i)}'] = components

    return part_components