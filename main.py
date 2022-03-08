from dataset.scene_generator import Scene
from utils import get_appearance_lens
from models.ransac import RANSAC
from models.gcm import GCM
from eval import eval
import pandas as pd
import argparse
import random
import json

def inference(
    num_faces : int = 3, dataset : str = 'test', plot : bool = True,
    sims : int = 100, restarts : int = 5, ll : float = 0.0001, seed : int = 1138, beta_0=0.0001):

    # Load geometry data.
    pose_orig  = pd.read_pickle('./dataset/labels/pose_orig.pkl')

    # Load appearance data.
    image_names = json.load(open('./dataset/labels/image_sets.json'))
    appearance_lens = get_appearance_lens(pd.read_pickle('./dataset/labels/appearance_labels.pkl'), image_names[dataset])

    appearance_gt = pd.read_pickle('./dataset/labels/appearance_labels.pkl')
    appearance_gt = appearance_gt[appearance_gt['name'].isin(image_names[dataset])][['name','eyes','nose','mouth','jaw','hair']]

    # Create scene generator.
    scene = Scene(
        appearance_gt=appearance_gt, appearance_lens=appearance_lens,
        pose_orig=pose_orig, transforms=None, beta_0=beta_0)

    # Load inference models. 
    gcm = GCM(sims, restarts)
    ransac = RANSAC()

    # Prepare face scene groups.
    set_size = appearance_gt.shape[0]
    indices = [idx for idx in range(set_size)]
    # random.seed(seed)
    random.shuffle(indices)

    counter=0
    while indices and len(indices)>=num_faces:

        # Generate the scene. 
        scene_indices = [indices.pop() for _ in range(num_faces)]
        print(f'=> scene indices: {scene_indices}')

        scene_data, face_names, generated_scenes, transforms, canvas_mask = scene.get_scene(face_indices=scene_indices)
        
        # Perform VI appearance and pose inference.
        ground_truth = generated_scenes[0]
        output = gcm.run(scene_data, lambda_0=ll, ground_truth=ground_truth, inference_index=counter, verbose=False)

        # Evaluate Appearance.
        print(f'VI correct assignment: {output["is_correct"]}')

        if output['is_correct']:
            # Evaluate results: Assignment accuracy and appearance.
            RMSE_VI = eval(output, generated_scenes, transforms, scene, face_names, scene_idx=counter, plot=plot)
            print(f'VI RMSE: {round(RMSE_VI,3)}')

        # Perform RANSAC appearance and pose inference.
        RMSE_RANSAC, is_correct = ransac.run(scene_data, scene_indices, ground_truth, counter, plot=plot)
        print(f'RANSAC correct assignment: {is_correct}')
        print(f'RANSAC RMSE: {round(RMSE_RANSAC,3)}')

        counter+=1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_faces', type=int,
                    help='The number of faces that will appear in the scene (choose 2-5)')
    
    args = parser.parse_args()

    inference(plot=True, num_faces=args.num_faces)
