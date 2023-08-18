import torch
import torch.nn.functional as F
import numpy as np
#from scipy.linalg.decomp import eig
import scipy
from scipy.linalg import eigh
from scipy import ndimage

import cupy as cp
from cupyx.scipy import ndimage as cuimage

#from sklearn.mixture import GaussianMixture
#from sklearn.cluster import KMeans

'''
    PyTorch-based codes
'''
def torch_linalg_eigh(A, B, subset_by_index):
    L = torch.linalg.cholesky(B)
    L_inv = torch.linalg.inv(L)
    C = torch.matmul(L_inv, torch.matmul(A, L_inv.transpose(-1,-2)))
    eigenvalues, eigenvectors = torch.linalg.eigh(C)
    subset_indices = slice(subset_by_index[0], subset_by_index[1] + 1)
    eigenvalues = eigenvalues[..., subset_indices]
    eigenvectors = torch.matmul(L_inv.transpose(-1,-2), eigenvectors[..., subset_indices])
    
    return eigenvalues, eigenvectors

def ncut(feats, dims, scales, init_image_size, tau = 0, eps=1e-5, im_name='', no_binary_graph=False):
    """
    Implementation of NCut Method.
    Inputs
      feats: the pixel/patche features of an image
      dims: dimension of the map from which the features are used
      scales: from image to map scale
      init_image_size: size of the image
      tau: thresold for graph construction
      eps: graph edge weight
      im_name: image_name
      no_binary_graph: ablation study for using similarity score as graph edge weight
    """
    feats = F.normalize(feats, p=2, dim=-2)
    A = (feats.transpose(-1,-2) @ feats)
    if no_binary_graph:
        A[A<tau] = eps
    else:
        A = A > tau
        A = torch.where(A.float() == 0, torch.tensor(eps).to(A.device), A)
    d_i = torch.sum(A, dim=-1)
    # D = torch.diag(d_i)
    e = torch.eye(d_i.shape[-1]).to(d_i.device)
    d_i_exp = d_i.unsqueeze(2).expand(*d_i.shape, d_i.shape[1])
    D = e * d_i_exp

    # Print second and third smallest eigenvector
    _, eigenvectors = torch_linalg_eigh(D-A, D, subset_by_index=[1,2])
    eigenvec = eigenvectors[..., 0].clone()


    # method1 avg
    second_smallest_vec = eigenvectors[..., 0]
    avg = torch.sum(second_smallest_vec, dim=-1) / second_smallest_vec.shape[-1]
    bipartition = second_smallest_vec > avg.unsqueeze(-1)

    seed = torch.argmax(torch.abs(second_smallest_vec), dim=-1)

    bool_factors = torch.gather(bipartition, 1, seed.unsqueeze(-1))
    bin_factors = bool_factors * 1
    bin_factors = torch.where(bin_factors==0, torch.tensor([-1]).to(bin_factors), bin_factors)

    # if bipartition[seed] != 1:
    #     eigenvec = eigenvec * -1
    #     bipartition = torch.logical_not(bipartition)
        
    eigenvec = eigenvec * bin_factors
    bipartition = torch.logical_xor(bipartition, torch.logical_not(bool_factors))
        
    bipartition = bipartition.reshape(dims).float()

    # predict BBox
    pred, _, objects,cc = detect_box(bipartition, seed, dims, scales=scales, initial_im_size=init_image_size) ## We only extract the principal object BBox
    mask = torch.zeros(dims).to(bipartition.device)
    mask[cc[0],cc[1],cc[2]] = 1

    # mask = torch.from_numpy(mask).to('cuda')
#    mask = torch.from_numpy(bipartition).to('cuda')
    bipartition = mask

    eigvec = second_smallest_vec.reshape(dims) 
    # eigvec = torch.from_numpy(eigvec).to('cuda')
    
    bipartition = F.interpolate(mask.unsqueeze(1), size=init_image_size, mode='nearest').squeeze()
    eigvec = F.interpolate(eigvec.unsqueeze(1), size=init_image_size, mode='nearest').squeeze()
    
    return  seed, bipartition, eigvec

def detect_box(bipartition, seed,  dims, initial_im_size=None, scales=None, principle_object=True):
    """
    Extract a box corresponding to the seed patch. Among connected components extract from the affinity matrix, select the one corresponding to the seed patch.
    """
    bs, w_featmap, h_featmap = dims
    objects, num_objects = cuimage.label(cp.asarray(bipartition))
    objects = torch.tensor(objects).to(seed.device)
    cc = objects[torch.arange(objects.shape[0]), seed.squeeze() // h_featmap, seed.squeeze() % h_featmap]

    if principle_object:
        # mask = cp.where(objects == cc)
        mask = torch.nonzero(torch.eq(objects, cc.unsqueeze(-1).unsqueeze(-1)), as_tuple=True)
        
        # # Add +1 because excluded max
        # ymin, ymax = min(mask[0]), max(mask[0]) + 1
        # xmin, xmax = min(mask[1]), max(mask[1]) + 1
        # # Rescale to image size
        # r_xmin, r_xmax = scales[1] * xmin, scales[1] * xmax
        # r_ymin, r_ymax = scales[0] * ymin, scales[0] * ymax
        # pred = [r_xmin, r_ymin, r_xmax, r_ymax]

        # # Check not out of image size (used when padding)
        # if initial_im_size:
        #     pred[2] = min(pred[2], initial_im_size[1])
        #     pred[3] = min(pred[3], initial_im_size[0])

        # # Coordinate predictions for the feature space
        # # Axis different then in image space
        # pred_feats = [ymin, xmin, ymax, xmax]
        
        pred, pred_feats = None, None

        return pred, pred_feats, objects, mask
    else:
        raise NotImplementedError

'''
    NumPy-based codes
'''
# def ncut(feats, dims, scales, init_image_size, tau = 0, eps=1e-5, im_name='', no_binary_graph=False):
#     """
#     Implementation of NCut Method.
#     Inputs
#       feats: the pixel/patche features of an image
#       dims: dimension of the map from which the features are used
#       scales: from image to map scale
#       init_image_size: size of the image
#       tau: thresold for graph construction
#       eps: graph edge weight
#       im_name: image_name
#       no_binary_graph: ablation study for using similarity score as graph edge weight
#     """
#     feats = F.normalize(feats, p=2, dim=0)
#     # A = (feats.transpose(0,1) @ feats)
#     A = (feats.transpose(-1,-2) @ feats)
#     A = A.cpu().numpy()
#     if no_binary_graph:
#         A[A<tau] = eps
#     else:
#         A = A > tau
#         A = np.where(A.astype(float) == 0, eps, A)
#     d_i = np.sum(A, axis=1)
#     D = np.diag(d_i)

#     # Print second and third smallest eigenvector
#     _, eigenvectors = eigh(D-A, D, subset_by_index=[1,2])
#     eigenvec = np.copy(eigenvectors[:, 0])


#     # method1 avg
#     second_smallest_vec = eigenvectors[:, 0]
#     avg = np.sum(second_smallest_vec) / len(second_smallest_vec)
#     bipartition = second_smallest_vec > avg

#     seed = np.argmax(np.abs(second_smallest_vec))

#     if bipartition[seed] != 1:
#         eigenvec = eigenvec * -1
#         bipartition = np.logical_not(bipartition)
#     bipartition = bipartition.reshape(dims).astype(float)

#     # predict BBox
#     pred, _, objects,cc = detect_box(bipartition, seed, dims, scales=scales, initial_im_size=init_image_size) ## We only extract the principal object BBox
#     mask = np.zeros(dims)
#     mask[cc[0],cc[1]] = 1

#     mask = torch.from_numpy(mask).to('cuda')
# #    mask = torch.from_numpy(bipartition).to('cuda')
#     bipartition = mask

#     eigvec = second_smallest_vec.reshape(dims) 
#     eigvec = torch.from_numpy(eigvec).to('cuda')
    
#     bipartition = F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=init_image_size, mode='nearest').squeeze()
#     eigvec = F.interpolate(eigvec.unsqueeze(0).unsqueeze(0), size=init_image_size, mode='nearest').squeeze()
    
#     return  seed, bipartition.cpu().numpy(), eigvec.cpu().numpy()

# def detect_box(bipartition, seed,  dims, initial_im_size=None, scales=None, principle_object=True):
#     """
#     Extract a box corresponding to the seed patch. Among connected components extract from the affinity matrix, select the one corresponding to the seed patch.
#     """
#     w_featmap, h_featmap = dims
#     objects, num_objects = ndimage.label(bipartition)
#     cc = objects[np.unravel_index(seed, dims)]


#     if principle_object:
#         mask = np.where(objects == cc)
#        # Add +1 because excluded max
#         ymin, ymax = min(mask[0]), max(mask[0]) + 1
#         xmin, xmax = min(mask[1]), max(mask[1]) + 1
#         # Rescale to image size
#         r_xmin, r_xmax = scales[1] * xmin, scales[1] * xmax
#         r_ymin, r_ymax = scales[0] * ymin, scales[0] * ymax
#         pred = [r_xmin, r_ymin, r_xmax, r_ymax]

#         # Check not out of image size (used when padding)
#         if initial_im_size:
#             pred[2] = min(pred[2], initial_im_size[1])
#             pred[3] = min(pred[3], initial_im_size[0])

#         # Coordinate predictions for the feature space
#         # Axis different then in image space
#         pred_feats = [ymin, xmin, ymax, xmax]

#         return pred, pred_feats, objects, mask
#     else:
#         raise NotImplementedError
