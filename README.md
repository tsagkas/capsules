# Inference for Generative Capsule Models
This code is the implementation of the algorithm described in the following paper: [Inference for Generative Capsule Models](https://arxiv.org/abs/2103.06676).

This repo contains the code for the "face experiments". The original constellation experiments can be found at: [anazabal/GenerativeCapsules](https://github.com/anazabal/GenerativeCapsules).

<p align="center">
<img alt="TSNE embeddings of object-capsule presence probabilities on MNIST digits." src=./results/demo.gif width="520" height="360">
<p align="center">
<b>Figure 1:</b> Reconstruction demo of randomly transformed faces in a given scene.
 </p>
</p>

# Instructions
## 1. Setup

a. Install manually the following dependecies:

  * `opencv-python`>=4.5.5.62
  * `scikit-learn`>=1.0.2
  * `matplotlib`>=3.5.1
  * `pandas`>=1.3.5
  * `monty`>=2022.1.19
  * `numpy`>=1.21.5


b. To generate the dataset run the following script:

    ./dataset/create_dataset.sh

**NOTE**: this step will take several hours since the code has to:
    
* generate 100,842 synthetic face images,
* train 5 PPCA models,
* train the FA model,
* generate 100,842 x 5 appearance labels. 

## 2. Running Experiments
To run the algorithm, select the number of faces that will exist in the scene and execute the following command:

    python -m main --num_faces=3

Results for our variational inference algorithm and the RANSAC-based approach will be saved in the `./results` directory.
## 3. Citation

## 4. Notes
The code was tested on Ubuntu 20.04.4 with python 3.7.10. All the part-images were downloaded from the [PhotoFitMe](https://www.open.edu/openlearn/PhotoFitMe) project page.


