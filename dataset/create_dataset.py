from models.pca import get_part_components, train_PCA
from dataset.face_generator import generate_face
from dataset.download_images import download_data
from matplotlib import pyplot as plt
import pickle as pk
import pandas as pd
import random
import json
import cv2
import os

"""
=======================
Dataset size:   100,842
-----------------------
Encoder:
    + train set: 60,000
    + val set:    7,614
Factor Analysis:
    + train set: 18,000
    + val set:    7,614
Global test set:
    + test set:   7,614
=======================
"""

def generate_images(show=False):
    counter = 0
    print('=> Generating dataset images..')

    for sex in ['female', 'male']:
        for race in ['asian', 'black', 'white']:
            for nose_idx in range(7):
                for eyes_idx in range(7):
                    for mouth_idx in range(7):
                        for jaw_idx in range(7):
                            for hair_idx in range(7):

                                # Create face image.
                                face = generate_face((sex, race, [jaw_idx, hair_idx, nose_idx, eyes_idx, mouth_idx]), show=False)

                                # Plot face image.
                                if show:
                                    plt.imshow(cv2.resize(face, (face.shape[0], face.shape[1])), cmap='gray')
                                    plt.show()

                                # Save face image.
                                dir_name = './dataset/faces'
                                if not os.path.exists(os.path.join(dir_name)):
                                    os.makedirs(os.path.join(dir_name))
                                cv2.imwrite(dir_name+'/face' + str(counter) + '.png', face)
                                
                                # Increase counter.
                                if counter%500 == 0:
                                    print(f"running: {round(100*counter/100842,2)}%", end = "\r")
                                counter += 1

    print('=> Done!')

def generate_labels():
    print('=> Generating dataset labels..')
    # Calculate the PCs for each part image
    # with the pre-trained PCA models.
    part_components = get_part_components()

    # Store for each synthetic face the 5 appearance
    # vectors (i.e. the PCs for each face part in the img)
    appearance_labels = {}
    part_names = ['name', 'jaw', 'hair', 'nose', 'eyes', 'mouth']
    for part_name in part_names:
        appearance_labels[part_name] = []

    counter = 0
    for sex in ['female', 'male']:
        for race in ['asian', 'black', 'white']:
            for nose_idx in range(7):
                for eyes_idx in range(7):
                    for mouth_idx in range(7):
                        for jaw_idx in range(7):
                            for hair_idx in range(7):
                                # Get each appearance vector for the specific face.
                                for part_name, idx in zip(part_names[1:], [jaw_idx, hair_idx, nose_idx, eyes_idx, mouth_idx]):
                                    appearance_labels[part_name].append(part_components[part_name][sex][race][f'{part_name}_{str(idx)}'][0])
                                appearance_labels['name'].append(f'face{str(counter)}.png')
                               
                                # Increase counter.
                                counter += 1
                                if counter%500 == 0:
                                    print(f"running: {round(100*counter/100842,2)}%", end = "\r")

    # Save face image.
    dir_name = './dataset/labels'
    if not os.path.exists(os.path.join(dir_name)):
        os.makedirs(os.path.join(dir_name))
    df = pd.DataFrame(appearance_labels)
    df.to_pickle(os.path.join(dir_name, 'appearance_labels.pkl'))
    
    print('=> Done!')

def split_data():
    NUM_SEXES, NUM_RACES = 2, 3
    NUM_PARTS, NUM_PART_TYPES = 7, 5
    CLASS_SIZE, NUM_CLASSES = NUM_PARTS**NUM_PART_TYPES, NUM_SEXES*NUM_RACES
    NUM_IMGS = CLASS_SIZE * NUM_SEXES * NUM_RACES

    img_names = [f'face{idx}.png' for idx in range(NUM_IMGS)]
    set_names = ['train_enc', 'train_fa', 'val_enc', 'val_fa', 'test']
    set_sizes  = [60000, 18000, 7614, 7614, 7614]
    sets = {}

    # Seperate images based on sex, race.
    img_names_per_class = {}
    class_names = ['fem_asian', 'fem_black', 'fem_white', 'male_asian', 'male_black', 'male_white']
    for idx, class_name in enumerate(class_names):
        img_names_per_class[class_name] = img_names[idx*CLASS_SIZE:(idx+1)*CLASS_SIZE]
        random.shuffle(img_names_per_class[class_name])

    # Create train, val and test sets, by sampling uniformly from all classes.
    for idx, set_name in enumerate(set_names):
        sets[set_name] = []
        for class_name in class_names:
            sets[set_name]+=[img_names_per_class[class_name].pop() for _ in range(set_sizes[idx]//NUM_CLASSES)]
    
    # Sort image names. 
    for set_name in set_names:
        sets[set_name] = sorted(sets[set_name], key=lambda x: float(x[4:-4]))

    # Test: Is original list empty?
    for class_name in class_names:
        assert img_names_per_class[class_name] == []
    # Test: Are there duplicates between the train, val, test sets?
    for set_name_1 in set_names:
        for set_name_2 in set_names:
            if set_name_1!=set_name_2:
                assert len(set(sets[set_name_1]).intersection(set(sets[set_name_2]))) == 0
    # Test: Are there duplicates in each set?
    for set_name in set_names:
        assert len(sets[set_name]) == len(set(sets[set_name]))

    # Save data sets.
    with open(f'./dataset/labels/image_sets.json', 'w') as outfile:
        json.dump(sets, outfile)

def main():
    # Download images from PhotoFitMe site.
    download_data()
    # Create all the possible synthetic faces.
    generate_images()
    # Train PCA models for the appearance vectors.
    train_PCA(percentage=0.95)
    # Create labels for dataset. 
    generate_labels()
    # Split to train, test.
    split_data()

if __name__ == '__main__':
    main()
