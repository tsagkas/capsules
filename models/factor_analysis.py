from sklearn.decomposition import FactorAnalysis
from dataset.face_generator import generate_face
from dataset.utils import get_face_mask
from utils import rmse
import matplotlib.pyplot as plt
import pandas as pd
import pickle as pk
import numpy as np
import json
import time
import cv2

def remove_pose(output):
    components = []
    indices = [0, 31, 48, 66, 87, 121]
    for idx, _ in enumerate(['eyes', 'nose', 'mouth', 'jaw', 'hair']):
        # Unit variance components.
        output_unit_var = output[:, indices[idx]:indices[idx+1]][:,6:] 
        components.append(output_unit_var)
    return np.hstack((components[0], components[1], components[2], components[3], components[4]))

def vector2dict(components):
    part_names = ['eyes', 'nose', 'mouth', 'jaw', 'hair']
    components_dict = {}
    for part_name in part_names:
        components_dict[part_name]=[]
    indices = [0, 24, 35, 47, 63, 91]
    for vect in components:
        for idx, name in enumerate(part_names):
            components_dict[name].append(vect[indices[idx]: indices[idx+1]])

    return components_dict

def dict2vector(sets, img_names):
    set = sets[sets['name'].isin(img_names)][['eyes','nose','mouth','jaw','hair']].to_numpy()
    appearance_len = set[0,0].shape[0]+set[0,1].shape[0]+set[0,2].shape[0]+set[0,3].shape[0]+set[0,4].shape[0]

    v = np.zeros((set.shape[0],appearance_len))
    for idx in range(set.shape[0]):
        v[idx,:] = np.hstack((set[idx,0],set[idx,1],set[idx,2],set[idx,3],set[idx,4]))
    return v

def latent2face(W, mu, num_factors, z=None, show=False):
    if z is None:
        # Sample random latent vector
        z = np.random.normal(size=num_factors)
    
    # Generate appearance.
    appearance = np.matmul(W,z) + mu #+error

    # Appearance vector => Appearance dict.
    components_dict = {}
    components_length = [0, 24, 11, 12, 16, 28]
    components_indices = [sum(components_length[:i+1]) for i in range(6)]

    for idx, part_name in enumerate(['eyes', 'nose', 'mouth', 'jaw', 'hair']):
        part_appearance = appearance[components_indices[idx]:components_indices[idx+1]]
        components_dict[part_name] = part_appearance

    # Generate face from appearance vector.
    face = generate_face(components=components_dict, show=show)

    return face

def tune_fa(output_pred):
    """
    Tune FA for different number of factors 
    (best results with 12 factors => pre-trained model at ./models/fa)
    """
    appearance_vecs  = pd.read_pickle('./dataset/labels/appearance_labels.pkl')
    image_names = json.load(open('./dataset/labels/image_sets.json'))

    train_set = dict2vector(appearance_vecs, image_names['train_fa'])
    val_set   = dict2vector(appearance_vecs, image_names['val_fa'])
    test_set  = dict2vector(appearance_vecs, image_names['test'])
    
    # RMSE
    errors = []
    # Tune num_factors.
    for num_factors in [2, 4, 6, 8, 12, 16, 24, 32]:
        print(f'=> k={num_factors}')

        # Train Factor Analysis model.
        fa = FactorAnalysis(n_components=num_factors)
        fa.fit(train_set)
        pk.dump(fa, open(f'./FA/fa_{num_factors}.pkl', 'wb'))

        #! Create model.
        W, mu = fa.components_.T, fa.mean_
        np.save(f'./FA/loadings_matrix_{num_factors}.npy', W)
        np.save(f'./FA/mu_vector_{num_factors}.npy', mu)
        
        # fa = pk.load(open(f'./FA/fa_{num_factors}.pkl', 'rb'))
        # W = np.load(f'./FA/loadings_matrix_{num_factors}.npy')
        # mu = np.load(f'./FA/mu_vector_{num_factors}.npy')
        
        loss = []
        latent_variables = fa.transform(test_set)

        skip=162
        counter=0
        # Evaluate predictions. 
        start = time.time()
        for idx, (z, img_name) in enumerate(zip(latent_variables, image_names['test'])):
            if idx%skip!=0:
                continue

            # generate face from latent code.
            face_generated = latent2face(W, mu, num_factors, z=z)

            # Load input face image.
            face_original = cv2.imread(f'./dataset/faces/{img_name}', 0)
            error = rmse(face_generated/255.0, face_original/255.0, get_face_mask(resize=True))
            loss.append(error)
            counter+=1

            # if not os.path.exists(f'./eval/tune_factors/{idx}'):
            #     os.makedirs(f'./eval/tune_factors/{idx}')
            # plt.imsave(f'./eval/tune_factors/{idx}/rface_{num_factors}.png', face_generated)
            # plt.imsave(f'./eval/tune_factors/{idx}/rface_og.png', face_original)

            # np.save(f'./eval/tune_factors/{idx}/rface_{num_factors}.npy', face_generated)
            # np.save(f'./eval/tune_factors/{idx}/rface_og.npy', face_original)

            # # output_dict = isolate_pcs(output_pred, part_idx=idx)
            # # print(output_dict)
            # output_dict = {}
            # for part_name in ['eyes', 'hair', 'jaw', 'mouth', 'nose']:
            #     output_dict[part_name]=appearance_vecs[part_name][int(img_name[4:-4])]
            # # print(output[idx].shape)
            # face_pca = generate_face(components=output_dict, show=False, normalize_face=False, normalize_parts=False)
            # np.save(f'./eval/tune_factors/{idx}/rface_pca.npy', face_pca)

        print(np.array(loss).mean())
        errors.append(loss)

    t = 7614//(skip*6)

    for idx, ll in enumerate(errors):
        print(f'=> k = {[2, 4, 6, 8, 12, 16, 24, 32][idx]}')
        print(f'\({round(np.mean(ll[0:t]),3)} \pm {round(np.std(ll[0:t]),3)}\)&\({round(np.mean(ll[t*1:t*2]),3)} \pm {np.round(np.std(ll[t*1:t*2]),3)}\)&\({round(np.mean(ll[t*2:t*3]),3)} \pm {np.round(np.std(ll[t*2:t*3]),3)}\)&\({round(np.mean(ll[t*3:t*4]),3)} \pm {np.round(np.std(ll[t*3:t*4]),3)}\)&\({round(np.mean(ll[t*4:t*5]),3)} \pm {np.round(np.std(ll[t*4:t*5]),3)}\)&\({round(np.mean(ll[t*5:t*6]),3)} \pm {np.round(np.std(ll[t*5:t*6]),3)}\)')
        print('TOTAL: ', round(np.mean(ll),3), round(np.std(ll),3))