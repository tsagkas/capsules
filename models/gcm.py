from monty.collections import AttrDict
from scipy.special import logsumexp
import numpy as np

np.seterr(all="ignore")

class GCM:
    def __init__(self, sims, restarts):
        self.sims = sims
        self.restarts = restarts  
        self.r_mnk = []

        self.last_assignment = None
    
    # Assignment matrix initialization (random)
    def _r_mnk_initialization(self):
        r_mnk = np.zeros((self.N, self.N))
        for i in range(self.K):
            for j in range(self.K):
                tmp = r_mnk[5*i:5*(i+1), 5*j:5*(j+1)]
                np.fill_diagonal(tmp, np.random.rand(5))
                r_mnk[5*i:5*(i+1), 5*j:5*(j+1)]=tmp
        # For occlusion.
        if self.M!=self.N:
            for idx in self.excluded_parts:
                r_mnk = np.delete(r_mnk, idx, 0)
        r_mnk = r_mnk / np.sum(r_mnk).reshape(-1, 1)
        return r_mnk

    # Extract the necessary information to perform inference
    def _unpack_scene_data(self, scene_data):
        self.X_m, self.F, self.N_k = scene_data['X_m'], scene_data['F'], scene_data['N_k']
        self.M, self.K, self.N = scene_data['M'], scene_data['K'], scene_data['N']
        self.D_kn, self.mu_kn = scene_data['D_kn'], scene_data['mu_kn']
        self.objects, self.visible_objects = scene_data['objects'], scene_data['visible_objects']

    # Set the hyperparameters of the model (mu_0, Lambda_0) for Y and lambda_0
    def _set_hyperparameters(self):
        self.mu_0 = np.zeros(12+4) 
        self.mu_0 = np.array([0,0,1,0]+[0]*12) 
        self.Lambda_0 = np.eye(12+4)
        self.a_mnk = np.eye(5*self.K)

        i_matrix, a_mnk = np.eye(5)/self.K, np.eye(5)/self.K
        for i in range(self.K-1):
            a_mnk = np.hstack((a_mnk,i_matrix))
        i_matrix, a_mnk = a_mnk, a_mnk
        for i in range(self.K-1):
            a_mnk = np.vstack((a_mnk,i_matrix))
        self.a_mnk = a_mnk

    # Initialize dict with the parameters that are necessary in each iteration
    def _params_initialization(self, rr):
        self.params = dict({
                'Lambda_k': [self.K*[self.Lambda_0]],
                'mu_k': [self.K*[self.mu_0]],
                'r_mnk': [self.r_mnk[rr]],
                'r_mnk_complete': [self.r_mnk[rr]],
                'lambda_0': [self.lambda_0],
                })

    # Reset model's params
    def _reinitialize(self, ELBO_epoch, lambda_0):
        r_mnk = self._r_mnk_initialization()
        self.params['r_mnk'][-1] = r_mnk
        self.params['r_mnk_complete'][-1] = r_mnk
        self.params['mu_k'][-1] = self.K*[self.mu_0]
        self.params['Lambda_k'][-1] = self.K*[self.Lambda_0]
        self.params['lambda_0'][-1] = lambda_0
        ELBO_epoch[-1] = -np.inf

    # Update the inferred transform's precision
    def _update_Lambda_k(self):
        r_mnk = self.params['r_mnk'][-1]
        Lambda_k = []
        for kk in range(self.K):
            result = np.zeros((16,16))
            for mm in range(self.M):
                for nn in range(self.N_k[kk]):
                    if self.X_m[mm].shape[0] != self.F[kk][nn].shape[0]:
                        continue
                    result += r_mnk[mm][nn+kk*5] * self.F[kk][nn].T@np.linalg.inv(self.D_kn[kk][nn]/self.lambda_0)@self.F[kk][nn]

            result += np.linalg.inv(self.Lambda_0)
            Lambda_k += [result]    
        self.params['Lambda_k'].append(Lambda_k)

    # Update the inferred transform
    def _update_mu_k(self):
        r_mnk = self.params['r_mnk'][-1]
        self.last_assignment = r_mnk
        Lambda_k_inv = np.linalg.inv(self.params['Lambda_k'][-1])
        mu_k = []

        for kk in range(self.K):
            result = np.zeros(16)
            for mm in range(self.M):
                for nn in range(self.N_k[kk]):
                    if self.X_m[mm].shape[0] != self.F[kk][nn].shape[0]:
                        continue

                    result += r_mnk[mm][nn+5*kk]*self.F[kk][nn].T@np.linalg.inv(self.D_kn[kk][nn]/self.lambda_0).T@(self.X_m[mm] - self.mu_kn[kk][nn])

            result += np.linalg.inv(self.Lambda_0).T@self.mu_0
            mu_k += [Lambda_k_inv[kk] @ result]

        self.params['mu_k'].append(mu_k)

    def _sinkhorn_knopp(self, log_r, max_iters=20, tol=1e-3):
        M, N = np.shape(log_r)
        sum_cols = logsumexp(log_r, axis=1)
        counter = 0
        while counter < max_iters and not (np.abs(np.exp(sum_cols[:M]) - 1) < tol).all():
            log_r = log_r - logsumexp(log_r, axis=0).reshape(1, -1)
            sum_cols = logsumexp(log_r, axis=1)
            log_r = log_r - sum_cols.reshape(-1, 1)
            counter += 1

        return log_r

    def _update_r_mnk(self, mahal_term, trace_term):
        # log_rho is log a_mnk for M = N
        log_rho = np.log(self.a_mnk)

        mu_k = self.params['mu_k'][-1]
        for kk in range(self.K):
            for mm in range(self.M):
                for nn in range(self.N_k[kk]):
                    if self.X_m[mm].shape[0] != self.F[kk][nn].shape[0]:
                        continue
                    log_rho[mm][nn+kk*5] =\
                         log_rho[mm][nn+kk*5]-0.5 * np.log(np.linalg.det(self.D_kn[kk][nn]/self.lambda_0)) \
                        - 0.5*self.X_m[mm].shape[0]*np.log(2*np.pi) \
                            -0.5*(self.X_m[mm]-self.F[kk][nn]@mu_k[kk]-self.mu_kn[kk][nn]).T@np.linalg.inv(self.D_kn[kk][nn]/self.lambda_0)@(self.X_m[mm]-self.F[kk][nn]@mu_k[kk]-self.mu_kn[kk][nn])\
                                -0.5*np.reshape(np.trace(self.F[kk][nn].T@np.linalg.inv(self.D_kn[kk][nn]/self.lambda_0)@self.F[kk][nn]@np.linalg.inv(self.params['Lambda_k'][-1][kk])), [1, -1])

        log_r = self._sinkhorn_knopp(log_rho, max_iters=2)

        self.params['r_mnk'].append(np.exp(log_r[:self.M, :]))
        self.params['r_mnk_complete'].append(np.exp(log_r))

    def _params_update(self):
        # Update Lambda_k
        self._update_Lambda_k()
        # Update mu_k
        self._update_mu_k()
        # Update assignment.
        self._update_r_mnk(mahal_term=None, trace_term=None)
        # Save current lambda
        self.params['lambda_0'].append(self.lambda_0)

        # Calculate the ELBO components. 
        r_mnk = self.params['r_mnk'][-1]
        mu_k = self.params['mu_k'][-1]
        E_log_p_x = 0
        for kk in range(self.K):
            for mm in range(self.M):
                for nn in range(self.N_k[kk]):

                    if self.X_m[mm].shape[0] != self.F[kk][nn].shape[0]:
                        continue

                    E_log_p_x += -r_mnk[mm][nn+kk*5]*0.5*self.X_m[mm].shape[0]*np.log(2*np.pi) -0.5*r_mnk[mm][nn+kk*5]*np.log(np.linalg.det(self.D_kn[kk][nn]/self.lambda_0)) \
                        -0.5*r_mnk[mm][nn+kk*5]*(self.X_m[mm]-self.F[kk][nn]@mu_k[kk]-self.mu_kn[kk][nn]).T@np.linalg.inv(self.D_kn[kk][nn]/self.lambda_0)@(self.X_m[mm]-self.F[kk][nn]@mu_k[kk]-self.mu_kn[kk][nn]) \
                            -0.5*r_mnk[mm][nn+kk*5]*np.trace(self.F[kk][nn].T@np.linalg.inv(self.D_kn[kk][nn]/self.lambda_0)@self.F[kk][nn]@np.linalg.inv(self.params['Lambda_k'][-1][kk]))
        KL_z = 0
        for kk in range(self.K):
            for mm in range(self.M):
                for nn in range(self.N_k[kk]):
                    KL_z += self.params['r_mnk_complete'][-1][mm][nn+kk*5] * (np.log(self.a_mnk[mm][nn+kk*5]+ 1e-9) - np.log(self.params['r_mnk_complete'][-1][mm][nn+kk*5] + 1e-9))

        KL_y = 0
        mu_k = self.params['mu_k'][-1]
        for kk in range(self.K):
            KL_y+= 0.5*np.log(np.linalg.det(np.linalg.inv(self.Lambda_0)))-0.5*np.log(np.linalg.det(self.params['Lambda_k'][-1][kk]))\
                -0.5*(mu_k[kk]-self.mu_0).T@np.linalg.inv(self.Lambda_0)@(mu_k[kk]-self.mu_0)-0.5*np.trace(np.linalg.inv(self.Lambda_0)@np.linalg.inv(self.params['Lambda_k'][-1][kk]))

        score = 0 
        for kk in range(self.K):
            for mm in range(self.M):
                for nn in range(self.N_k[kk]):
                    if self.X_m[mm].shape[0] != self.F[kk][nn].shape[0]:
                        continue
                    score += (self.X_m[mm]-self.F[kk][nn]@mu_k[kk]-self.mu_kn[kk][nn]).T@np.linalg.inv(self.D_kn[kk][nn]/self.lambda_0)@(self.X_m[mm]-self.F[kk][nn]@mu_k[kk]-self.mu_kn[kk][nn]) * self.params['r_mnk'][-1][mm][nn+kk*5]
        return [E_log_p_x, KL_y, KL_z], score

    def _stop(self, ELBO_epoch, new_ELBO, lambda_0=0.0001):
        lambda_max = 1

        tol = 1e-3
        ELBO = ELBO_epoch[-1]

        index_k = np.concatenate([np.zeros(1), np.cumsum(self.N_k)]).astype(int)
        if np.abs(ELBO - new_ELBO) < tol:
            # Sparsity Constraint: reset assignments if object assigned with less than 2 points
            for kk in range(self.K):
                perm_matrix = self.params['r_mnk'][-1][:,index_k[kk]:index_k[kk+1]]
                if perm_matrix.max() > 0.9 and perm_matrix.sum() < 2:
                    self.lambda_0 = lambda_0
                    self._reinitialize(ELBO_epoch, lambda_0)
                    print('Wrong assignment!')
                    return False

            #Continue if lambda hasn't reached its max value
            if self.lambda_0 < lambda_max:
                self.lambda_0 *= 10
                return False
            else:
                return True
        else:
            return False

    def generate_objects(self, transform):
        return [F@y for F, y in zip(transform, self.F)]

    def arrange_output(self, final_params):
        r_mnk=final_params['r_mnk'][-1] 
        transform=final_params['mu_k'][-1]
        
        order = []
        zero_counter = 0
        for kn in range(self.K):
            for km in range(self.K):
                if (np.round(r_mnk[km*5:(km+1)*5,kn*5:(kn+1)*5])==np.eye(5)).all():
                    order.append(km)
                else:
                    zero_counter+=1

        if len(order)==self.K and zero_counter==self.K**2-self.K:
            assignment = True
        else:
            assignment = False
        if assignment:
            final_params['mu_k'][-1]= [x for _, x in sorted(zip(order,transform))]

        return assignment, final_params
   
    # Metric based on determinant of r_mnk to determine correctness
    def _is_correct(self):
        return self.is_doubly_stochastic()

    def is_doubly_stochastic(self):
        for mm in range(self.M):
            if not np.sum(np.round(self.params['r_mnk'][-1][mm]))==1:
                return False

        if not np.sum(np.round(self.params['r_mnk'][-1]))==self.M:
            return False

        if np.logical_or(np.round(self.params['r_mnk'][-1],1)==0.0, np.round(self.params['r_mnk'][-1],1)==1.0).all():
            return True
        else:
            return False 

    def run(self, scene_data, lambda_0, ground_truth, inference_index, verbose=True):
        self.lambda_0_org = lambda_0
        self.lambda_0 = lambda_0
        self.ground_truth = ground_truth
        self.inference_index=inference_index

        # Extract data from given scene
        self._unpack_scene_data(scene_data)

        for _ in range(self.restarts):
            self.r_mnk.append(self._r_mnk_initialization())
        
        # Test different restarts
        final_ELBO = -np.inf
        for rr in range(self.restarts):
            self._set_hyperparameters()

            # Initialize dict with all variables necessary for the inference
            self._params_initialization(rr)
            self.params['r_mnk'][-1] = self._r_mnk_initialization()

            # VBEM
            new_ELBO = -np.inf
            ELBO_epoch = [new_ELBO]
            results = []

            self.lambda_0=0.0001

            for ss in range(self.sims):
                #Update parameters and get ELBO
                ELBO_terms, score = self._params_update()
                ELBO = np.sum(ELBO_terms)

                # Save results
                results.append([ELBO] + ELBO_terms + [score])

                if verbose and not ss % 1:
                    print(f'ss: {ss} -- ELBO: {ELBO}, log_x: {ELBO_terms[0]}, KL_Y: {-ELBO_terms[1]}, KL_Z: {-ELBO_terms[2]}, Score: {score}')

                # stopping criteria
                ELBO_epoch += [ELBO]
                if self._stop(ELBO_epoch, new_ELBO, self.lambda_0):
                    break
                else:
                    new_ELBO = ELBO

            # Check whether the current restart is better than the previous one
            if ELBO > final_ELBO:
                final_ELBO = ELBO
                final_params = self.params

            # Not to be used. 
            if self._is_correct():
                break

        correct, final_params = self.arrange_output(final_params)

        output = AttrDict(
            assignment=final_params['r_mnk'][-1], 
            transform=final_params['mu_k'][-1],
            is_correct=correct,
            stats={'rr':rr, 'ss':ss})

        return output