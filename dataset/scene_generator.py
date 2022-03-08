from dataset.utils import get_face_mask
from utils import pixel2cartesian
from dataclasses import dataclass
from typing import List, Dict
import pandas as pd
import pickle as pk
import numpy as np
import warnings 
import random
import cv2

@dataclass
class Part:
    # Part name: ['eyes','nose','mouth','jaw','hair']
    part_name : str
    # Appearance in principal subspace.
    appearance_gt : np.array
    # Similarity transformation vec. 
    pose_gt : np.array
    pose_orig : np.array

@dataclass
class Face:
    # In this case, it's just 'face'.
    obj_name : str 
    # List of Part objects.
    parts : List
    # GT geometric transform (applied).
    yk : np.array
    # Face appearance latent code. 
    z : np.array = None

class Scene:
    def __init__(
        self, 
        appearance_gt : pd.DataFrame, appearance_lens : List, 
        pose_orig : pd.DataFrame, transforms : pd.DataFrame, 
        num_factors : int = 12, beta_0 : float = 0.0001) -> None:

        # Part names. 
        self.part_names = ['eyes','nose','mouth','jaw','hair']

        # The size of each appearance in its principal subspace.
        self.appearance_lens = appearance_lens
        
        # GT appearance and pose of parts.
        self.appearance_gt = appearance_gt
        self.pose_orig = pose_orig 

        # Appearance.
        self.W = np.load(f'./models/fa/loadings_matrix_{num_factors}.npy')
        self.mu = np.load(f'./models/fa/mu_vector_{num_factors}.npy')
        self.beta_0 = beta_0

        #* Get the submatrics of the loadings matrix.
        self._get_appearance_matrices()

    def _get_appearance_matrices(self):
        self.appearance_matrices = {}
        total_len = 0
        for idx, part_name in enumerate(self.part_names):
            appearance_len = self.appearance_lens[idx]

            self.appearance_matrices[part_name] = self.W[total_len:total_len+appearance_len,:]
            total_len+=appearance_len

    def _get_F_matrices(self, parts : List[Part]) -> List[np.array]:
        Fkn = []
        
        for idx, part in enumerate(parts):
            x, y, s_cos, s_sin = part.pose_orig
            F_matrix = np.zeros((self.appearance_lens[idx]+4, 4+12))
            F_matrix[0:4, 0:4] = np.array([
                [1, 0, x,  y], 
                [0, 1, y, -x],
                [0, 0, s_cos, -s_sin], 
                [0, 0, s_sin,  s_cos]
            ])

            F_matrix[4:, 4:] = self.appearance_matrices[part.part_name]
            Fkn.append(F_matrix)

        return Fkn

    def _get_feature_vecs(self, parts : List[Part]) -> List[np.array]:
        X_m = []
        for part in parts:
            X_m.append(np.hstack((part.pose_gt, part.appearance_gt)))
        return X_m

    def get_scene(self, face_indices : List) -> Dict:
        # observed pose/appearance vectors.
        X_m = []
        # observed faces in the scene. 
        faces = []
        generated_scenes, poses, transforms, canvas_mask = generate_scene([self.appearance_gt.iloc[idx].to_dict()['name'] for idx in face_indices])
        
        for i, idx in enumerate(face_indices):
            # Get the ground truth and predicted appearance vector.
            appearance_dict_gt = self.appearance_gt.iloc[idx].to_dict()

            # Get the ground truth and predicted pose vector.
            pose_dict_gt = poses[i]
            pose_dict_orig = self.pose_orig.iloc[0].to_dict()

            # Get the applied face geometric transform.
            self.yk = transforms[i]

            # Create the 5 face-parts. 
            parts = []
            for part_name in self.part_names:
                parts.append(Part(
                    part_name=part_name, 
                    appearance_gt=appearance_dict_gt[part_name],
                    pose_gt=pose_dict_gt[part_name],
                    pose_orig=pose_dict_orig[part_name]
                    ))

            # Create face from parts.
            faces.append(Face(
                obj_name=appearance_dict_gt['name'], 
                parts=parts, 
                yk=self.yk))

            # Store observed pose/appearance vectors.
            X_m += self._get_feature_vecs(faces[-1].parts)

        # Get templates 
        Fkn = self._get_F_matrices(faces[-1].parts)
        indices = np.concatenate([np.zeros(1), np.cumsum(self.appearance_lens)]).astype(int)

        # Load the factor analysis model and get the Psi matrix. 
        fa_model = pk.load(open(f'./models/fa/fa_{12}.pkl', 'rb'))
        psi = fa_model.noise_variance_

        # Create the diagonal variance matrix and the mean vector. 
        D_kn=[[np.diag([self.beta_0]*4+psi[indices[idx]:indices[idx+1]].tolist()) for idx in range(5)] for _ in range(len(face_indices))]
        mu_kn =  np.array([[[0]*4 + self.mu[indices[idx]:indices[idx+1]].tolist() for idx in range(5)] for _ in range(len(face_indices))], dtype=object)

        return { 'D_kn': D_kn, 'mu_kn':mu_kn,
                'K': len(face_indices), 'M':len(X_m), 'N':5*len(face_indices), 
                'X_m':X_m, 'F':[Fkn]*len(face_indices), 'N_k':[5]*len(face_indices),
                'objects':['face']*len(face_indices), 'visible_objects':np.ones(len(face_indices))
                }, [f.obj_name for f in faces], generated_scenes, transforms, canvas_mask

def similarity_transform(transforms, img_size=224):
    """
    Transform Object Templates.
    """
    img_org_size=640
    part_templates = { # (rows, cols)
    'eyes' : (int((190+84)*(img_size/img_org_size)), img_size/2), 
    'nose' : (int((270+84)*(img_size/img_org_size)), img_size/2), 
    'mouth': (int((340+84)*(img_size/img_org_size)), img_size/2), 
    'hair' : (int(100*(img_size/img_org_size)), img_size/2), 
    'jaw'  : (int(497*(img_size/img_org_size)), img_size/2)}

    observed_points = []
    for transform in transforms:
        part_poses = {}
        # Get transformation parameters.
        t_x, t_y, scale, theta = transform
        # Create the transformation vector.
        transform = np.array([t_x/(img_size/2), -t_y/(img_size/2),
            scale * np.cos(np.radians(theta)), scale * np.sin(np.radians(theta))])
            
        # Turn 2D coordinates into templates.
        for part_name, value in part_templates.items():
            row, col = value
            x, y = pixel2cartesian(row, col, img_size)
            template = np.array([[1, 0, x, y], [0, 1, y, -x]])
            # Transform templates.
            x, y = template @ transform
            part_poses[part_name] = [x, y, scale*np.cos(np.radians(theta)), scale*np.sin(np.radians(theta))]
        observed_points.append(part_poses)

    return observed_points

def generate_scene(face_names : List):
    # Get the face mask for placing the faces on the background.
    face_mask = get_face_mask(resize=True)

    input_faces = []
    for face_name in face_names:
        input_faces.append(cv2.imread(f'./dataset/faces/{face_name}', 0))

    radius, counter =112, 0

    # Loop until a plausible scene is found.
    while True:
        counter+=1
        transforms = []
        scene_check = True

        # NOTE: scales are fixed to speed scene generation.
        scales = [0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15]
        for idx, _ in enumerate(input_faces):
            t_x = random.choice([-80+i*10 for i in range(17)])
            t_y = random.choice([-80+i*10 for i in range(17)])
            scale = scales[idx]
            theta = random.choice([i for i in range(360)])

            transforms.append([t_x, t_y, scale, theta])

        def l2_distance(c1, c2):
            return ((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)**(0.5)

        # Check if the faces are overlapping.
        for tt1 in range(len(transforms)):
            for tt2 in range(tt1+1, len(transforms)):
                c1, c2 = transforms[tt1], transforms[tt2]
                dist = l2_distance([c1[0], c1[1]], [c2[0], c2[1]])

                if dist <= radius*c1[2] + radius*c2[2]:
                    scene_check=False
                    break

        # Check if the faces are inside the canvas.
        for tt in transforms:
            x, y = tt[0], tt[1]
            scale = tt[2]
            if abs(x)+scale*radius>radius or abs(y)+scale*radius>radius:#abs(y)+scale*radius>radius
                scene_check=False
                break

        if scene_check:
            break

        if counter>10000:
            warnings.warn(f'The process is taking too long to find a plausible scene! Consider decreasing the number of faces!')

    canvas = np.zeros((224,224))
    canvas_mask = np.zeros((224,224))
    for idx, input_face in enumerate(input_faces):
        face = face_mask*input_face

        t_x, t_y, scale, theta = transforms[idx]

        R = cv2.getRotationMatrix2D(((224-1)/2.0,(224-1)/2.0), theta, scale)
        T = np.float32([[1, 0, t_x],[0, 1, t_y]])

        face=cv2.warpAffine(face, R, (224,224))
        face=cv2.warpAffine(face, T, (224,224))

        transformed_mask = cv2.warpAffine(get_face_mask(resize=True), R, (224,224))
        transformed_mask = cv2.warpAffine(transformed_mask, T, (224,224))

        canvas_mask += transformed_mask
        canvas+=face

    # Place faces on the canvas.
    canvas_mask[canvas_mask>1]=1
    canvas_mask = np.ones((224,224))-canvas_mask
    generated_scenes = [canvas_mask*100+canvas]

    # Get the new poses of the face parts. 
    poses = similarity_transform(transforms)

    return generated_scenes, poses, transforms, np.ones((224,224))-canvas_mask