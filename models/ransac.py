from models.factor_analysis import latent2face 
from dataset.utils import get_face_mask 
from utils import rmse, get_angle_diff
import matplotlib.pyplot as plt
from utils import pose2params
from models.pca import pc2img
import pickle as pk
import numpy as np
import copy
import cv2

class RANSAC():
    def __init__(self) -> None:
        # FA: Loadings matrix and mean vector -- Object templates. 
        self.W = np.load('./models/fa/loadings_matrix_12.npy')
        self.mu = np.load('./models/fa/mu_vector_12.npy')
        # Load the factor analysis model and get the Psi matrix. 
        fa_model = pk.load(open(f'./models/fa/fa_{12}.pkl', 'rb'))
        psi = fa_model.noise_variance_
        self.psi_diag = psi
        self.Psi = np.zeros((91,91))
        np.fill_diagonal(self.Psi, psi)

    def _generate_images(self, X_m):
        part_imgs, part_poses = [], []

        for idx, x_m in enumerate(X_m):
            if x_m is not None:
                # get part name.
                part_name = self.part_names[idx%5]

                # get part image. 
                part_img  = pc2img(x_m[4:], part_name)
                part_imgs.append(part_img)

                # get part pose. 
                t_x, t_y, theta, scale = pose2params(x_m[:4])
                part_poses.append([t_x, t_y, theta, scale])
            else:
                part_imgs.append(None)
                part_poses.append(None)

        return part_imgs, part_poses

    def run(self, scene_data, scene_indices, ground_truth, counter, plot):
        # Setting.
        self.part_names = ['eyes', 'nose', 'mouth', 'jaw', 'hair']
        self.vec_lengths = [24, 11, 12, 16, 28]

        # Get observed data and part templates. 
        X_m = scene_data['X_m']
        F_kns = scene_data['F'][0]

        # Additional scene info:
        num_objects, num_observer_parts = scene_data['K'], scene_data['M']
        W_idx = [[ 0, 24], [24, 35], [35, 47], [47, 63], [63, 91]]

        lens = [24, 11, 12, 16, 28]
        indices = np.concatenate([np.zeros(1), np.cumsum(lens)]).astype(int)

        X = copy.deepcopy(X_m)
        if type(X) != list:
            X = X.tolist()

        correct_assignment=[[i for i in range(num_objects)] for _ in range(5)]

        part_imgs, part_poses = self._generate_images(X)
        assignments, errors, transforms = [], [], []
        for offset in range(num_objects*5):
            r_mnk = [[-1 for _ in range(num_objects)] for _ in range(5)]
            subset_transforms = []

            # Choose part to find concensus (bruteforce).
            for idx, x_m in enumerate(X):
                if r_mnk[idx%5][idx//5] != -1 or idx%5<offset-1 or x_m is None:
                    continue

                # Infer geometric transform: y_k^g.
                F_kn = F_kns[idx%5]
                y_kg = np.linalg.pinv(F_kn)@x_m

                # Infer appearance transform: y_k^a.
                W_v = self.W[W_idx[idx%5][0]:W_idx[idx%5][1], :]
                Psi_inv = np.linalg.inv(self.Psi)
                Psi_v = Psi_inv[W_idx[idx%5][0]:W_idx[idx%5][1],W_idx[idx%5][0]:W_idx[idx%5][1]]
                Cov_z = np.linalg.inv(np.eye(12) + W_v.T@Psi_v@W_v)
                mu_z  = Cov_z@W_v.T@Psi_v@(x_m[4:]-self.mu[indices[idx%5]:indices[idx%5+1]])
                y_ka = np.random.multivariate_normal(mu_z, Cov_z)

                # Store transform.
                subset_transforms.append(np.hstack((y_kg[:4],y_ka)))

                # Assign the main part. 
                r_mnk[idx%5][idx//5] = idx//5

                # Generate parts using the inferred transform:
                part_a, part_g = [], [] # part appearances and geometries. 
                for idy, part_name in enumerate(self.part_names):
                    # Appearance. 
                    x_a = F_kns[idy][4:,4:]@y_ka+self.mu[indices[idy]:indices[idy+1]]
                    part_a.append(pc2img(x_a, part_name))
                    # Geometry.
                    x_g = F_kns[idy]@y_kg
                    part_g.append([pose2params(x_g[:4])])

                # Find sub-match.
                loss = [[np.inf for _ in range(num_objects)] for _ in range(5)]
                for idy, xx in enumerate(X): 
                    if r_mnk[idy%5][idy//5] != -1 or xx is None:
                        continue

                    # Tested part's appearance and pose.
                    img_test = part_imgs[idy]
                    t_x_test, t_y_test, theta_test, scale_test = part_poses[idy]

                    # Inferred part's pose
                    img_inferred = part_a[idy%5]
                    t_x, t_y, theta, scale = part_g[idy%5][0]

                    # Check if pose matches.
                    if np.abs(t_x_test-t_x)<0.05 and np.abs(t_y_test-t_y)<0.05 and\
                        np.abs(scale_test-scale)<0.05 and get_angle_diff(theta_test,theta)<5:
                        # Calculate error.
                        RMSE = rmse(img_inferred, img_test)
                        # If better error make assignment. 
                        if loss[idy%5][idy//5] > RMSE:
                            if idy//5 not in r_mnk[idy%5]:
                                loss[idy%5][idy//5] = RMSE 
                                r_mnk[idy%5][idy//5] = idx//5 

            assignments.append(r_mnk)
            errors.append(sum([sum(ll) for ll in loss]))
            transforms.append(subset_transforms)

        best_index = errors.index(min(errors))

        best_transform  = transforms[best_index]
        best_assignment = assignments[best_index]
        best_error      = errors[best_index]

        correct_assignment=[[i for i in range(num_objects)] for _ in range(5)]

        is_correct = True if best_assignment==correct_assignment else False

        scene_mask = np.zeros((224,224)) 
        generated_scene = np.zeros((224,224))

        for mu_k in best_transform:
            mask_f = get_face_mask(resize=True)
            x_f, y_f, theta_f, s_f = pose2params(mu_k[:4], pixel_domain=True)
            y_appearance = mu_k[4:]
            generated_face = latent2face(W=self.W, mu=self.mu, num_factors=None, z=y_appearance, show=False)

            M_f = cv2.getRotationMatrix2D(((224-1)/2.0,(224-1)/2.0),theta_f, s_f)
            rotated_fake = cv2.warpAffine(generated_face, M_f, (224,224))
            mask_f = cv2.warpAffine(mask_f, M_f, (224,224))

            M_f = np.float32([[1,0,x_f],[0,1,y_f]])
            trans_fake = cv2.warpAffine(rotated_fake, M_f, (224,224))
            mask_f = cv2.warpAffine(mask_f, M_f, (224,224))
            
            fake = trans_fake*mask_f
            scene_mask+= mask_f
            mask_f = np.ones((224,224))-mask_f
            generated_scene=generated_scene*mask_f
            generated_scene+= fake

        scene_mask[scene_mask>1]=1
        masks = np.ones((224,224)) - scene_mask
        generated_scene += masks*100

        if plot:
            _, axarr = plt.subplots(1,2)

            axarr[1].imshow(generated_scene, cmap='gray', vmin=0, vmax=255)
            axarr[1].set_xlabel(f'Generated Scene', fontsize=12)
            axarr[1].axes.xaxis.set_ticks([])
            axarr[1].axes.yaxis.set_ticks([])

            axarr[0].imshow(ground_truth, cmap='gray', vmin=0, vmax=255)
            axarr[0].set_xlabel(f'Input Scene', fontsize=12)
            axarr[0].axes.xaxis.set_ticks([])
            axarr[0].axes.yaxis.set_ticks([])

            plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
            plt.savefig(f'./results/result_RANSAC_{counter}')
            plt.show()

        return rmse(ground_truth/255.0, generated_scene/255.0), is_correct