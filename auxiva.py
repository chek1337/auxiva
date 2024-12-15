import torch as t
import enum
from torch import nn
from typing import Optional

# In this model, we consider a determined situation, so c = k = s 

# Wh has a dimension (f x k x c)
# where f - frequency
#       k - kiosk :)
#       c - channel 
# Matrix element - wh^f_kc

# V has a dimension (k x k x k x f) // (k{1} x k{2} x k{3} x f)
# where k - kiosk
#       f - frequency
# Matrix element - Vk^f_kk          // Vk{1}^f_k{2}k{3}

# X has a dimension (c x f x b)
# where c - channel
#       f - frequency
#       b - bin/batch
# Matrix element - x^c_fb

# Y has a dimension (s x f x b)
# where s - source
#       f - frequency
#       b - bin/batch
# Matrix element - y^s_fb

# All examples are written for c/s/k = 2, f = 3, b = 4

def phi_laplace(r, _):
    return 1/(2*r)

def phi_gauss(r, N_freq):
    return N_freq/(r**2)

# https://arxiv.org/pdf/2009.05288
def projection_back(Wh, Y): 
    scale = t.linalg.inv(Wh)[:, :, 0]
    return t.einsum("sfb,fs->sfb",  Y, scale)

#
# The usual multiplication of a matrix by a vector
# 
# |wh^1_11 wh^1_12| \/ |x^1_11 x^1_12 x^1_13 x^1_14| 
# |wh^1_21 wh^1_22| /\ |x^2_11 x^2_12 x^2_13 x^2_14|
#
# |wh^2_11 wh^2_12| \/ |x^1_21 x^1_22 x^1_23 x^1_24|  _   размерность
# |wh^2_21 wh^2_11| /\ |x^2_21 x^2_22 x^2_23 x^2_24|  ‾   (s x f x b)
#
# |wh^3_11 wh^3_12| \/ |x^1_31 x^1_32 x^1_33 x^1_34| 
# |wh^3_21 wh^3_22| /\ |x^2_31 x^2_32 x^2_33 x^2_34| 
#
def demix(Wh, X):
    return t.einsum("fsc,cfb->sfb", Wh, X)

#
# Take the scalar norm of N_freq on each bin 
# (s x f x b) -> (s x 1 x b) -> (s x b) *without preserving the dimension*
#
def calc_r(Y):
    fisrt = t.norm(Y, dim=1) # By frequency
    return fisrt

#
# 1) Multiply x*x^h
#
#   this x_a                            this x_b
# |/*x^1_11*/ x^1_12 x^1_13 x^1_14|   |/*x^1_11*/ x^1_12 x^1_13 x^1_14|  
# |  x^1_21   x^1_22 x^1_23 x^1_24|   |  x^1_11   x^1_12 x^1_13 x^1_14|  
# |  x^1_31   x^1_32 x^1_33 x^1_34|   |  x^1_11   x^1_12 x^1_13 x^1_14|  
#                                   x         
# |/*x^2_11*/ x^2_12 x^2_13 x^2_14|   |/*x^1_11*/ x^1_12 x^1_13 x^1_14|  
# |  x^2_31   x^2_32 x^2_33 x^2_34|   |  x^1_11   x^1_12 x^1_13 x^1_14|  
# |  x^2_21   x^2_22 x^2_23 x^2_24|   |  x^1_11   x^1_12 x^1_13 x^1_14|  
#            (c x f x b)                         (c x f x b)
#
# In fact, I will take the selected elements
# x_a * x_b^H = |x^1_11| \/ |\bar{x^1_11}, \bar{x^1_11}| _ |a  b| _  some 
#               |x^2_11| /\                              ‾ |c  d|   matrix
# If you do this for all f and b, you will get a resulting matrix of size (c x c x f x b)
# 
def calc_XXh(X):
    return t.einsum("i...,j...->ij...", X, X.conj())

#
# 1) Multiply phi(r_k) of dimension (1 x b) by the result of the first action
# It turns out that first[:, :, :, b] * phi(r_k)
# That is, each bin is multiplied by its own coefficient
#
# 1.1) Take the average (math expectation) among all bin's
# The result is a tensor of V dimension (с x c x c x f) 
#
def calc_V(V, XXh, r, N_freq, phi):
    for k in range(r.shape[0]):
        V[k, :, :, :]= t.mean(phi(r[k], N_freq)*XXh, dim=3)

#
# 1) Multiply W_h of dimension (f x k x c) by Vk of dimension (k x k x f)
#
# |wh^1_11 wh^1_12| \/ |Vk^1_11 Vk^1_12| 
# |wh^1_21 wh^1_22| /\ |Vk^1_21 Vk^1_22|
#
# |wh^2_11 wh^2_12| \/ |Vk^2_11 Vk^2_12|  _    dimension
# |wh^2_21 wh^2_22| /\ |Vk^2_21 Vk^2_22|  ‾   (f x k x k)
#
# |wh^3_11 wh^3_12| \/ |Vk^3_11 Vk^3_12| 
# |wh^3_21 wh^3_22| /\ |Vk^3_21 Vk^3_22|
#    (f x k x c )         (k x k x f)
# 
# 2) Take the inverse matrix for each frequency f
#
# 3.1) Multiply each matrix by a by e_k (I take the kth column)
# |a^1_11 a^2_12| \/ |1|
# |a^1_21 a^2_22| /\ |0|
#
# |a^2_11 a^2_12| \/ |1|  _  размерность
# |a^2_21 a^2_22| /\ |0|  ‾    (f x k)
#
# |a^3_11 a^3_12| \/ |1|
# |a^3_21 a^3_22| /\ |0|
#  ( f x k x k )   
#
# 3.2) since I worked with Wh, and matrix updates take into account w_k, 
# that (Wh_k)^h = w_k
#
def calc_update1(Wh, V_k, k):
    first = t.einsum("fik,kjf->fij", Wh, V_k)
    second = t.linalg.inv(first)
    third = second[:, :, k].conj()
    return third

#
# 1.1) Multiply the string by the matrix
#
# |wh^1_11 wh^1_12| \/ |Vk^1_11 Vk^1_12|     |a b|
#                   /\ |Vk^1_21 Vk^1_22|
#
# |wh^2_11 wh^2_12| \/ |Vk^2_11 Vk^2_12|  _  |c d|
#                   /\ |Vk^2_21 Vk^2_22|  ‾     
#
# |wh^3_11 wh^3_12| \/ |Vk^3_11 Vk^3_12|     |e f|
#                   /\ |Vk^3_21 Vk^3_22|
#      (f x c)            (k x k x f)       (f x c)
# 
# 1.2) Multiply rows by columns
#            w_k=(wh_k)^h
# |a b| \/ |\bar{wh^1_11}|
#       /\ |\bar{wh^1_12}|
#         
# |c d| \/ |\bar{wh^2_11}|  _    |i j k|
#       /\ |\bar{wh^2_12}|  ‾    (1 x f)
#
# |e f| \/ |\bar{wh^3_11}|
#       /\ |\bar{wh^3_12}|
#
# 2) We take the root from 
# sqrt(|i j k|) = |x y z| 
#-----------------------------------------------------------------------------------
# Then, with the second update, multiply the vector w_k for each frequency by its coefficient obtained in (1)
#
# |\bar{wh^1_11}|  / \/
# |\bar{wh^1_12}| /  /\
#         
# |\bar{wh^2_11}|  / |_|  _ dimension 
# |\bar{wh^2_12}| /   _|  ‾  (f x c)
#
# |\bar{wh^3_11}|  /  ‾/
# |\bar{wh^3_12}| /   /_
#     (f x c)
#
def calc_update2(Wh_k, V_k):
    first = t.einsum('fc,ckf,fk->f', Wh_k, V_k, Wh_k.conj())
    second = t.sqrt(first).view(-1,1)
    return second

class Model(enum.Enum):
    LAPLACE = "laplace"
    GAUSS = "gauss"

class auxiva(nn.Module):
    def __init__(
        self,
        n_iter: int,
        model: Optional[Model] = Model.LAPLACE
    ):
        super().__init__()
        self.n_iter = n_iter
        if model == "gauss":
            self.phi = phi_gauss
        elif model == "laplace":
            self.phi = phi_laplace
        else:
            raise NotImplementedError("Unknown contrast function.")

    def forward(self, X) -> t.Tensor:
        K, N_freq, N_bins =  X.shape
        XXh = calc_XXh(X)
        Y = t.zeros_like(X).type_as(X)
        Wh = t.eye(K).unsqueeze(0).repeat(N_freq, 1, 1).type_as(X)
        V = t.zeros((K, K, K, N_freq)).type_as(X)

        for iter in range(self.n_iter):
            Y = demix(Wh, X) 
            r = calc_r(Y)
            calc_V(V, XXh, r, N_freq, self.phi)
            for k in range(K):
                Wh[:, k, :] = calc_update1(Wh, V[k, :, :, :], k)
                Wh[:, k, :] /= calc_update2(Wh[:, k, :], V[k, :, :, :])
        
        Y = demix(Wh, X) 
        
        Y = projection_back(Wh, Y)
        
        return Y
        
class bssSepation(nn.Module):
    def __init__(
        self,
        win_size: int,
        hop_length: Optional[int] = None,
        window_fun: Optional[t.tensor] = None,
        n_iter: Optional[int] = 20,
        model: Optional[Model] = Model.LAPLACE
    ):
        super().__init__()

        self.win_size = win_size

        if hop_length is None:
            self.hop_length = win_size//2
        else:
            self.hop_length = hop_length

        if window_fun is None:
            self.window_fun = t.hann_window(win_size)
        elif window_fun.size(0) == win_size:
            self.window_fun = window_fun
        else:
            raise ValueError(f"Size of window_fun {window_fun.shape} not 1D or not equal to win_size.")
        
        self.model = model

        self.n_iter = n_iter

        self.separator = auxiva(n_iter, model)
    
    def forward(self, mixed_audio) -> t.Tensor:
        with t.no_grad():
            X = t.stft(input=mixed_audio, 
                    n_fft=self.win_size, 
                    hop_length=self.hop_length, 
                    window=self.window_fun, 
                    return_complex=True)
            
            Y = self.separator(X)

            demixed_audio = t.istft(input=Y, 
                                    n_fft=self.win_size, 
                                    hop_length=self.hop_length, 
                                    window=self.window_fun)
            return demixed_audio