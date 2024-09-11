# Stain Reconstruction Augmentation (SRA)

import numpy as np
from PIL import Image
import random
import csv
import torch
import adaptive_mats

adaptive_matrix                 = True   # Set to true to load stain separation matrix from current slide, otherwise a global fixed stain separation matrix will be loaded which has lower accuracy.
dataset                         = 'tcga' # Options: 'tcga' for TCGA KIRC, 'ukidney' for Utah KIRC

random_mat_option               = 0      # 0=no random (use original stain separation mat), we always choose 0 in our paper; 1=randomly take an existing mat; 2=generate a random mat
bypass_H_E_coeffs               = False  # Bypass augmenting/tuning strength of H stain and E stain
random_chance_pureHE            = 0.1    # 0.0(bypass), 0.1=(5% pureH, 5% pureE) 
add_residual_channel            = 0      # 0=ignore residual channel (we always choose 0 in our paper), 1=add residual with alpha distribution, 2=use original from mat
add_random_beta_bias            = False  # Range: -0.05 ---0.05
add_traditional_H_E_coeffs      = False  # To mimic traditional stain augmentation (TSA) (ref: doi:10.1117/12.2293048)
#additional_norm_factor = 2.0

beta_low         = -.05                  # default = -.05
beta_high        = .05                   # default = .05
alpha_res_low    = .95                   # default = .95
alpha_res_high   = 1.05                  # default = 1.05

if add_traditional_H_E_coeffs:
    alpha_H_low  = 0.95                  # default = .95
    alpha_H_high = 1.05                  # default = 1.05
    alpha_E_low  = 0.95                  # default = .95
    alpha_E_high = 1.05                  # default = 1.05

HERes2RGBabsorp_matrices, H_range, E_range = adaptive_mats(dataset)
def take_random_existing_HERes2RGBabsorp_mat():
    random_index                            = np.random.randint(0, HERes2RGBabsorp_matrices.shape[0])
    HERes2RGBabsorp_random_existing_matrice = HERes2RGBabsorp_matrices[random_index]
    return HERes2RGBabsorp_random_existing_matrice

def generate_random_HERes2RGBabsorp_mat():
    # generate random H vector (row 1 of HERes2RGBabsorp matrice)
    random_H_vector = np.array([np.random.uniform(low, high) for low, high in H_range])
    random_H_L2norm_vector = np.linalg.norm(random_H_vector)
    random_H_vector = random_H_vector / random_H_L2norm_vector # L2 normalization

    # generate random E vector (row 1 of HERes2RGBabsorp matrice)
    random_E_vector        = np.array([np.random.uniform(low, high) for low, high in E_range])
    random_E_L2norm_vector = np.linalg.norm(random_E_vector)
    random_E_vector        = random_E_vector / random_E_L2norm_vector # L2 normalization

    # generate random Residual vector (row 1 of HERes2RGBabsorp matrice)
    Res_vector      = np.cross(random_H_vector, random_E_vector) # cross product
    Res_vector_norm = np.linalg.norm(Res_vector)
    Res_vector      = Res_vector/Res_vector_norm

    HERes2RGBabsorp_random_existing_matrice = np.column_stack((random_H_vector, random_E_vector, Res_vector))


    return HERes2RGBabsorp_random_existing_matrice

class rgb_he_wrgb:
    def __init__(self, Hmax_dist_type='beta', Emax_dist_type='beta', Hmax_dist_params=(0.0, 1.0, 0.0, 0.0), Emax_dist_params=(0.0, 1.0, 0.0, 0.0)):
        self.Hmax_dist_type   = Hmax_dist_type
        self.Hmax_dist_params = Hmax_dist_params
        self.Emax_dist_type   = Emax_dist_type
        self.Emax_dist_params = Emax_dist_params

    def generate_H_max_coeff(self):
        if self.Hmax_dist_type == 'uniform':
            return random.uniform(*self.Hmax_dist_params[:2])
        elif self.Hmax_dist_type == 'normal':
            return random.gauss(*self.Hmax_dist_params[:2])
        elif self.Hmax_dist_type == 'binomial':
            return np.random.binomial(*self.Hmax_dist_params[:2])
        elif self.Hmax_dist_type == 'trinomial':
            p_0, p_1 = self.Hmax_dist_params[:2]
            rand_num = random.random()
            if rand_num < p_0:
                return 0
            elif rand_num < p_0 + p_1:
                return 1
            else:
                return 0.5
        elif self.Hmax_dist_type == 'beta':
            alpha, beta, loc, scale = self.Hmax_dist_params
            beta_dist = torch.distributions.Beta(torch.tensor([alpha]), torch.tensor([beta]))
            sample = beta_dist.sample().item()
            transformed_sample = sample * scale + loc
            return transformed_sample
        else:
            raise ValueError("Unsupported distribution type!")
        
    def generate_E_max_coeff(self):
        if self.Emax_dist_type == 'uniform':
            return random.uniform(*self.Hmax_dist_params[:2])
        elif self.Emax_dist_type == 'normal':
            return random.gauss(*self.Hmax_dist_params[:2])
        elif self.Emax_dist_type == 'binomial':
            return np.random.binomial(*self.Hmax_dist_params[:2])
        elif self.Emax_dist_type == 'trinomial':
            p_0E, p_1E = self.Emax_dist_params[:2]
            rand_num = random.random()
            if rand_num < p_0E:
                return 0
            elif rand_num < p_0E + p_1E:
                return 1
            else:
                return 0.5
        elif self.Emax_dist_type == 'beta':
            alpha, beta, loc, scale = self.Emax_dist_params
            beta_dist               = torch.distributions.Beta(torch.tensor([alpha]), torch.tensor([beta]))
            sample                  = beta_dist.sample().item()
            transformed_sample      = sample * scale + loc
            return transformed_sample
        else:
            raise ValueError("Unsupported distribution type!")
        
    def generate_random_alpha_residual(self):
        return random.uniform(alpha_res_low, alpha_res_high)
        
    def __call__(self, image, adaptive_params):
        # Use the provided WSI name to get the corresponding pre-calculated stain separation matrix, H_max, E_max. 
        if adaptive_params is not None and adaptive_matrix:
            self.background_RGB      = adaptive_params['background_RGB']
            self.RGBabsorp2H         = adaptive_params['RGB_absorption_to_H']
            self.RGBabsorp2E         = adaptive_params['RGB_absorption_to_E']
            self.RGBabsorp2Res       = adaptive_params['RGB_absorption_to_Res']
            self.HERes2RGBabsorp_mat = adaptive_params['Mat_HERes2RGBabsorp']
            self.maxH                = adaptive_params['maxH']
            self.maxE                = adaptive_params['maxE']
            #print(f'background_RGB: {self.background_RGB}, RGB_absorption_to_H: {self.RGBabsorp2H}, RGB_absorption_to_E: {self.RGBabsorp2E}, , maxH: {self.maxH},, maxE: {self.maxE}')
        else:
            print('**** WARNING: No Adaptive Params Received by rgb_he_wrgb transform. Using default params... ****')
            self.background_RGB      = np.array([255.0, 255.0, 255.0])
            self.RGBabsorp2H         = np.array([1.58354329, -0.20067344, 0.13555159])
            self.RGBabsorp2E         = np.array([-1.15938766, 1.05805232, 0.32031071])
            self.RGBabsorp2Res       = np.array([0.129653323,-0.41473296, 0.90065898])
            self.HERes2RGBabsorp_mat = np.array([[0.67779000, 0.07773051, -0.12965332],
                                                 [0.62591322, 0.90127937, -0.41473296],
                                                 [0.38578927, 0.42620824,  0.90065897]])
            self.maxH                = 0.5
            self.maxE                = 0.5

        if isinstance(image, Image.Image):
            image = np.asarray(image)
        image      = image.astype('float32')
        image_ones = np.ones(image.shape, dtype=np.float32)
        image      = np.maximum(image, image_ones)

        # Convert RGB images into Optical Density (OD) space
        RGB_absorption = np.log10((image_ones / image) * self.background_RGB)

        # Use stain separation matrix to get H channel images and E channel images
        gray_image_H   = np.dot(RGB_absorption, self.RGBabsorp2H)
        #gray_image_H   = gray_image_H / (self.maxH * additional_norm_factor)
        #gray_image_H   = np.clip(gray_image_H, 0.0, 1.0)

        gray_image_E   = np.dot(RGB_absorption, self.RGBabsorp2E)
        #gray_image_E   = gray_image_E / (self.maxE * additional_norm_factor)
        #gray_image_E   = np.clip(gray_image_E, 0.0, 1.0)

        if add_residual_channel>0:
            gray_image_Res = np.dot(RGB_absorption, self.RGBabsorp2Res)
        
        H_max_coeff = self.generate_H_max_coeff()#calculate the distribution (mean, std) of H_max, then generate this random number. 
        #H_max_coeff = np.clip(H_max_coeff, .5, 2.0)  

        E_max_coeff = self.generate_E_max_coeff()
        #E_max_coeff = np.clip(E_max_coeff, .2, 2.0)  

        if add_residual_channel==1:
            Res_alpha = self.generate_random_alpha_residual()

        if random_chance_pureHE>0.0:
            rand_pure_H_E=np.random.uniform(0.0, 1.0)
            if rand_pure_H_E<random_chance_pureHE/2.0:
                H_max_coeff=0.0
                gray_image_H=0.0*gray_image_H
            elif rand_pure_H_E<random_chance_pureHE:
                E_max_coeff=0.0
                gray_image_E=0.0*gray_image_E

        # --- Hmax: min=.72, max=1.08 ; Emax: min=.29, max=.91

        if add_traditional_H_E_coeffs and not bypass_H_E_coeffs:
            alpha_H      = np.random.uniform(alpha_H_low, alpha_H_high)
            alpha_E      = np.random.uniform(alpha_E_low, alpha_E_high)
            gray_image_H = alpha_H*gray_image_H
            gray_image_E = alpha_E*gray_image_E
        elif not add_traditional_H_E_coeffs and not bypass_H_E_coeffs:
            gray_image_H = (H_max_coeff/self.maxH)*gray_image_H
            gray_image_E = (E_max_coeff/self.maxE)*gray_image_E            

        if add_residual_channel==1:
            gray_image_Res = Res_alpha*gray_image_Res #***

        if add_random_beta_bias:
            beta_H=np.random.uniform(beta_low, beta_high)
            beta_E=np.random.uniform(beta_low, beta_high)
            if add_residual_channel>0:
                beta_Res=np.random.uniform(beta_low, beta_high)
            
            gray_image_H = gray_image_H + beta_H
            gray_image_H=np.clip(gray_image_H, a_min=0.0, a_max=None)

            gray_image_E = gray_image_E + beta_E
            gray_image_E = np.clip(gray_image_E, a_min=0.0, a_max=None)

            if add_residual_channel>0:
                gray_image_Res = gray_image_Res+beta_Res
                gray_image_Res = np.clip(gray_image_Res, a_min=0.0, a_max=None)

        # Recombine the H and E images to create color images on OD space (OD = Optical Density)
        if random_mat_option==0:
            color_image_H_OD = np.expand_dims(gray_image_H, axis=-1) * self.HERes2RGBabsorp_mat[:, 0]
            color_image_E_OD = np.expand_dims(gray_image_E, axis=-1) * self.HERes2RGBabsorp_mat[:, 1]
            if add_residual_channel>0:
                color_image_Res_OD = np.expand_dims(gray_image_Res, axis=-1) * self.HERes2RGBabsorp_mat[:, 2]
        elif random_mat_option==1:
            rand_HERes2RGBmat=take_random_existing_HERes2RGBabsorp_mat()
            color_image_H_OD = np.expand_dims(gray_image_H, axis=-1) * rand_HERes2RGBmat[:, 0]
            color_image_E_OD = np.expand_dims(gray_image_E, axis=-1) * rand_HERes2RGBmat[:, 1]
            if add_residual_channel>0:
                color_image_Res_OD = np.expand_dims(gray_image_Res, axis=-1) * rand_HERes2RGBmat[:, 2]
        else:
            rand_HERes2RGBmat=generate_random_HERes2RGBabsorp_mat()
            color_image_H_OD = np.expand_dims(gray_image_H, axis=-1) * rand_HERes2RGBmat[:, 0]
            color_image_E_OD = np.expand_dims(gray_image_E, axis=-1) * rand_HERes2RGBmat[:, 1]
            if add_residual_channel>0:
                color_image_Res_OD = np.expand_dims(gray_image_Res, axis=-1) * rand_HERes2RGBmat[:, 2]

        color_image_HE_OD = color_image_H_OD + color_image_E_OD

        if add_residual_channel>0:
            color_image_HE_OD += color_image_Res_OD

        # Inverse of the logarithmic transformation
        # Convert image from OD space back to RGB space
        color_image_HE = self.background_RGB / (10 ** color_image_HE_OD)

        # Normalize and convert to uint8
        color_image_HE = np.clip(color_image_HE, 0, 255).astype(np.uint8)
        return Image.fromarray(color_image_HE)
