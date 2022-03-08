from models.factor_analysis import latent2face
from dataset.utils import get_face_mask
from utils import rmse, pose2params
import matplotlib.pyplot as plt
from typing import List
import numpy as np
import cv2

def eval(output, canvas, y_k, scene, face_names : List, scene_idx : int, plot : bool = False) -> float:
    num_faces = len(face_names)
    
    # Generate faces from inferred appearance vectors. 
    generated_faces = []
    for fidx in range(num_faces):
        y_appearance = output['transform'][fidx][4:]
        generated_faces.append(latent2face(W=scene.W, mu=scene.mu, num_factors=None, z=y_appearance, show=False))

    # Evaluate the appearance and pose of the generated scene w/ RMSE. 
    RMSE = 0
    scene_mask = np.zeros((224,224)) 
    generated_scene = np.zeros((224,224))

    for fidx in range(num_faces):

        mask_f = get_face_mask(resize=True)
        x_f, y_f, s_f, theta_f = y_k[fidx]
        x_f, y_f, theta_f, s_f = pose2params(output['transform'][fidx][0:4], pixel_domain=True)

        M_f = cv2.getRotationMatrix2D(((224-1)/2.0,(224-1)/2.0),theta_f, s_f)
        rotated_fake = cv2.warpAffine(generated_faces[fidx], M_f, (224,224))
        mask_f = cv2.warpAffine(mask_f, M_f, (224,224))

        M_f = np.float32([[1,0,x_f],[0,1,y_f]])
        trans_fake = cv2.warpAffine(rotated_fake, M_f, (224,224))
        mask_f = cv2.warpAffine(mask_f, M_f, (224,224))

        fake = trans_fake*mask_f
        generated_scene+= fake
        scene_mask+= mask_f

    masks = np.ones((224,224)) - scene_mask
    generated_scene += masks*100

    RMSE += rmse(generated_scene/255.0, canvas[0]/255.0)

    if plot:
        _, axarr = plt.subplots(1,2)

        ground_truth = canvas[0]
        axarr[1].imshow(generated_scene, cmap='gray', vmin=0, vmax=255)
        axarr[1].set_xlabel(f'Generated Scene', fontsize=12)
        axarr[1].axes.xaxis.set_ticks([])
        axarr[1].axes.yaxis.set_ticks([])

        axarr[0].imshow(ground_truth, cmap='gray', vmin=0, vmax=255)
        axarr[0].set_xlabel(f'Input Scene', fontsize=12)
        axarr[0].axes.xaxis.set_ticks([])
        axarr[0].axes.yaxis.set_ticks([])

        plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
        plt.savefig(f'./results/result_VI_{scene_idx}')
        plt.show()

    return RMSE
