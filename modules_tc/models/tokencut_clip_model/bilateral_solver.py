from scipy.sparse import diags
from scipy.sparse.linalg import cg
from scipy.sparse import csr_matrix
import numpy as np
from skimage.io import imread
from scipy import ndimage
import PIL.Image as Image 

'''
    PyTorch-based codes
'''
import torch
import torch.nn.functional as F
import cupy as cp
from cupyx.scipy import ndimage as cuimage
from cupyx.scipy.sparse import csr_matrix as cp_csr_matrix
from cupyx.scipy.sparse import diags as cp_diags
from cupyx.scipy.sparse.linalg import cg as cp_cg

RGB_TO_YUV = torch.tensor([
    [ 0.299,     0.587,     0.114],
    [-0.168736, -0.331264,  0.5],
    [ 0.5,      -0.418688, -0.081312]])
YUV_TO_RGB = torch.tensor([
    [1.0,  0.0,      1.402],
    [1.0, -0.34414, -0.71414],
    [1.0,  1.772,    0.0]])
YUV_OFFSET = torch.tensor([0, 128.0, 128.0]).reshape(1, 1, -1)
MAX_VAL = 255.0
def rgb2yuv(im):
    return torch.tensordot(im, RGB_TO_YUV.to(im.device), ([3], [1])) + YUV_OFFSET.to(im.device)

def yuv2rgb(im):
    return torch.tensordot(im.float() - YUV_OFFSET.to(im.device), YUV_TO_RGB.to(im.device), ([3], [1]))


def get_valid_idx(valid, candidates):
    """Find which values are present in a list and where they are located"""
    locs = torch.searchsorted(valid, candidates)
    # Handle edge case where the candidate is larger than all valid values
    locs = torch.clip(locs, 0, valid.shape[-1] - 1)
    # Identify which values are actually present
    valid_idx = torch.nonzero(torch.gather(valid, -1, locs) == candidates)
    locs = locs[valid_idx]
    # locs = locs[valid_idx[:,0], valid_idx[:,1]]
    
    return valid_idx, locs

class BilateralGrid(object):
    def __init__(self, im, sigma_spatial=32, sigma_luma=8, sigma_chroma=8):
        self.bs = im.shape[0]
        self.device = im.device
        im_yuv = rgb2yuv(im)
        # Compute 5-dimensional XYLUV bilateral-space coordinates
        Iy, Ix = torch.meshgrid(torch.arange(im.shape[1]), torch.arange(im.shape[2]))
        Ix, Iy = Ix.to(im.device), Iy.to(self.device)
        Ix = Ix.expand([self.bs, im.shape[1], im.shape[2]])
        Iy = Iy.expand([self.bs, im.shape[1], im.shape[2]])
        x_coords = (Ix / sigma_spatial).int()
        y_coords = (Iy / sigma_spatial).int()
        luma_coords = (im_yuv[..., 0] /sigma_luma).int()
        chroma_coords = (im_yuv[..., 1:] / sigma_chroma).int()
        coords = torch.concat((x_coords.unsqueeze(-1), y_coords.unsqueeze(-1), luma_coords.unsqueeze(-1), chroma_coords), dim=-1)
        coords_flat = coords.reshape(self.bs, -1, coords.shape[-1])
        _, self.npixels, self.dim = coords_flat.shape
        # Hacky "hash vector" for coordinates,
        # Requires all scaled coordinates be < MAX_VAL
        self.hash_vec = (MAX_VAL**torch.arange(self.dim).to(self.device))
        # self.hash_vec = ((MAX_VAL//4)**torch.arange(self.dim).to(self.device))
        # Construct S and B matrix
        self._compute_factorization(coords_flat)
        
    def _compute_factorization(self, coords_flat):
        # Hash each coordinate in grid to a unique value
        hashed_coords = self._hash_coords(coords_flat)
        
        unique_hashes, idx, counts = torch.unique(hashed_coords, dim=1, sorted=True, return_inverse=True, return_counts=True)
        _, ind_sorted = torch.sort(idx, stable=True)
        cum_sum = counts.cumsum(0)
        cum_sum = torch.cat((torch.tensor([0]).to(self.device), cum_sum[:-1]))
        unique_idx = ind_sorted[cum_sum]    
        
        # Identify unique set of vertices
        unique_coords = coords_flat[:, unique_idx]
        self.nvertices = unique_coords.shape[1]
        # Construct sparse splat matrix that maps from pixels to vertices
        # self.S = cp_csr_matrix((torch.ones(self.npixels).to(self.device), (idx, torch.arange(self.npixels))))
        sparse_tensor = torch.sparse_coo_tensor(torch.stack((idx, torch.arange(self.npixels).to(self.device))), torch.ones(self.npixels).to(self.device))
        self.S = sparse_tensor.to_dense()
        
        # Construct sparse blur matrices.
        # Note that these represent [1 0 1] blurs, excluding the central element
        self.blurs = []
        for i in range(self.bs):
            blurs = []
            for d in range(self.dim):
                blur = 0.0
                for offset in (-1, 1):
                    offset_vec = torch.zeros((1, self.dim)).to(self.device)
                    offset_vec[:, d] = offset
                    neighbor_hash = self._hash_coords(unique_coords[i] + offset_vec)                
                    valid_coord, idx = get_valid_idx(unique_hashes[i], neighbor_hash)
                    sparse_tensor = torch.sparse_coo_tensor(torch.stack((valid_coord.squeeze(-1), idx.squeeze(-1))), torch.ones((len(valid_coord),)).to(self.device), size=(self.nvertices, self.nvertices))
                    blur = blur + sparse_tensor.to_dense()
                    
                blurs.append(blur)
            blurs = torch.stack(blurs)
            self.blurs.append(blurs)
        self.blurs = torch.stack(self.blurs)
        
    def _hash_coords(self, coord):
        """Hacky function to turn a coordinate into a unique value"""
        return torch.matmul(coord.float(), self.hash_vec)
        # return torch.matmul(coord.to(torch.float64), self.hash_vec.to(torch.float64))

    def splat(self, x):
        return torch.matmul(self.S, x.float())
        # return torch.matmul(self.S, x)
    
    def slice(self, y):
        # return torch.matmul(self.S.T, torch.tensor(y).to(self.S.device))
        return torch.matmul(self.S.T, y)
    
    def blur(self, x):
        """Blur a bilateral-space vector with a 1 2 1 kernel in each dimension"""
        assert x.shape[-1] == self.nvertices
        out = 2 * self.dim * x
        # for blur in self.blurs:
        #     out = out + torch.matmul(blur, x)
        for i in range(self.dim):
            if x.dim() == 2:
                out = out + torch.matmul(self.blurs[:,i,...], x.unsqueeze(-1)).squeeze(-1)
            elif x.dim() == 3:
                out = out + torch.matmul(self.blurs[:,i,...], x)
            else:
                assert False
        return out

    def filter(self, x):
        """Apply bilateral filter to an input x"""
        return self.slice(self.blur(self.splat(x))) /  \
               self.slice(self.blur(self.splat(cp.ones_like(x))))

def bistochastize(grid, maxiter=10):
    """Compute diagonal matrices to bistochastize a bilateral grid"""
    m = grid.splat(torch.ones(grid.npixels).to(grid.device)).expand((grid.bs, -1))
    n = torch.ones(grid.bs, grid.nvertices).to(grid.device)
    for i in range(maxiter):
        n = torch.sqrt(n * m / grid.blur(n))
    # Correct m to satisfy the assumption of bistochastization regardless
    # of how many iterations have been run.
    m = n * grid.blur(n)
    Dm = torch.diag_embed(m)
    Dn = torch.diag_embed(n)
    return Dn, Dm

class BilateralSolver(object):
    def __init__(self, grid, params):
        self.grid = grid
        self.params = params
        self.Dn, self.Dm = bistochastize(grid)
    
    def solve(self, x, w):
        # Check that w is a vector or a nx1 matrix
        if w.ndim == 2:
            assert(w.shape[1] == 1)
        elif w.dim == 1:
            w = w.reshape(w.shape[0], 1)
        A_smooth = (self.Dm - torch.matmul(self.Dn, self.grid.blur(self.Dn)))
        w_splat = self.grid.splat(w)
        A_data = torch.diag_embed(w_splat[:,:,0])
        A = self.params["lam"] * A_smooth + A_data
        xw = x * w
        b = self.grid.splat(xw)
        # Use simple Jacobi preconditioner
        A_diag = torch.maximum(A.permute(1,2,0).diagonal(), torch.ones(A.permute(1,2,0).diagonal().shape).to(A.device)*self.params["A_diag_min"])
        M = torch.diag_embed(1/A_diag)
        # Flat initialization
        y0 = self.grid.splat(xw) / w_splat
        yhat = torch.empty_like(y0) 
        assert x.shape[-1] == 1       
        yhat = torch.linalg.solve(A, b)
        xhat = self.grid.slice(yhat)
        
        import math
        if math.isnan(xhat.sum()):
            print("NaN")
        
        return xhat

def bilateral_solver_output(input_images, target, sigma_spatial = 24, sigma_luma = 4, sigma_chroma = 4) : 
    
    # reference = np.array(Image.open(img_pth).convert('RGB'))
    bs, h, w = target.shape
    confidence = torch.ones((bs, h, w)).to(target.device) * 0.999
    
    grid_params = {
        'sigma_luma' : sigma_luma, # Brightness bandwidth
        'sigma_chroma': sigma_chroma, # Color bandwidth
        'sigma_spatial': sigma_spatial # Spatial bandwidth
    }

    bs_params = {
        'lam': 256, # The strength of the smoothness parameter
        'A_diag_min': 1e-5, # Clamp the diagonal of the A diagonal in the Jacobi preconditioner.
        'cg_tol': 1e-5, # The tolerance on the convergence in PCG
        'cg_maxiter': 25 # The number of PCG iterations
    }

    grid = BilateralGrid(input_images.permute(0,2,3,1), **grid_params)

    t = target.reshape(grid.bs, -1, 1).double()
    c = confidence.reshape(grid.bs, -1, 1).double()
    
    ## output solver, which is a soft value
    output_solver = BilateralSolver(grid, bs_params).solve(t, c).reshape(grid.bs, h, w)
    
    binary_solver = cuimage.binary_fill_holes(cp.asarray(output_solver)>0.5)
    labeled, nr_objects = cuimage.label(binary_solver)

    labeled = torch.tensor(labeled).to(input_images.device)
    nb_pixel = torch.tensor([torch.sum(labeled == i) for i in range(nr_objects + 1)]).to(target.device)
    pixel_order = torch.argsort(nb_pixel)
    
    binary_solver = torch.tensor(binary_solver).to(input_images.device)
    
    try : 
        binary_solver = labeled == pixel_order[-2]
    except: 
        binary_solver = torch.ones((bs, h, w), dtype=bool).to(input_images.device)
    
    return output_solver, binary_solver


'''
    NumPy-based codes
'''
# RGB_TO_YUV = np.array([
#     [ 0.299,     0.587,     0.114],
#     [-0.168736, -0.331264,  0.5],
#     [ 0.5,      -0.418688, -0.081312]])
# YUV_TO_RGB = np.array([
#     [1.0,  0.0,      1.402],
#     [1.0, -0.34414, -0.71414],
#     [1.0,  1.772,    0.0]])
# YUV_OFFSET = np.array([0, 128.0, 128.0]).reshape(1, 1, -1)
# MAX_VAL = 255.0
# def rgb2yuv(im):
#     return np.tensordot(im, RGB_TO_YUV, ([2], [1])) + YUV_OFFSET

# def yuv2rgb(im):
#     return np.tensordot(im.astype(float) - YUV_OFFSET, YUV_TO_RGB, ([2], [1]))


# def get_valid_idx(valid, candidates):
#     """Find which values are present in a list and where they are located"""
#     locs = np.searchsorted(valid, candidates)
#     # Handle edge case where the candidate is larger than all valid values
#     locs = np.clip(locs, 0, len(valid) - 1)
#     # Identify which values are actually present
#     valid_idx = np.flatnonzero(valid[locs] == candidates)
#     locs = locs[valid_idx] 
#     return valid_idx, locs

# class BilateralGrid(object):
#     def __init__(self, im, sigma_spatial=32, sigma_luma=8, sigma_chroma=8):
#         im_yuv = rgb2yuv(im)
#         # Compute 5-dimensional XYLUV bilateral-space coordinates
#         Iy, Ix = np.mgrid[:im.shape[0], :im.shape[1]]
#         x_coords = (Ix / sigma_spatial).astype(int)
#         y_coords = (Iy / sigma_spatial).astype(int)
#         luma_coords = (im_yuv[..., 0] /sigma_luma).astype(int)
#         chroma_coords = (im_yuv[..., 1:] / sigma_chroma).astype(int)
#         coords = np.dstack((x_coords, y_coords, luma_coords, chroma_coords))
#         coords_flat = coords.reshape(-1, coords.shape[-1])
#         self.npixels, self.dim = coords_flat.shape
#         # Hacky "hash vector" for coordinates,
#         # Requires all scaled coordinates be < MAX_VAL
#         self.hash_vec = (MAX_VAL**np.arange(self.dim))
#         # Construct S and B matrix
#         self._compute_factorization(coords_flat)
        
#     def _compute_factorization(self, coords_flat):
#         # Hash each coordinate in grid to a unique value
#         hashed_coords = self._hash_coords(coords_flat)
#         unique_hashes, unique_idx, idx = \
#             np.unique(hashed_coords, return_index=True, return_inverse=True) 
#         # Identify unique set of vertices
#         unique_coords = coords_flat[unique_idx]
#         self.nvertices = len(unique_coords)
#         # Construct sparse splat matrix that maps from pixels to vertices
#         self.S = csr_matrix((np.ones(self.npixels), (idx, np.arange(self.npixels))))
#         # Construct sparse blur matrices.
#         # Note that these represent [1 0 1] blurs, excluding the central element
#         self.blurs = []
#         for d in range(self.dim):
#             blur = 0.0
#             for offset in (-1, 1):
#                 offset_vec = np.zeros((1, self.dim))
#                 offset_vec[:, d] = offset
#                 neighbor_hash = self._hash_coords(unique_coords + offset_vec)
#                 valid_coord, idx = get_valid_idx(unique_hashes, neighbor_hash)
#                 blur = blur + csr_matrix((np.ones((len(valid_coord),)),
#                                           (valid_coord, idx)),
#                                          shape=(self.nvertices, self.nvertices))
#             self.blurs.append(blur)
        
#     def _hash_coords(self, coord):
#         """Hacky function to turn a coordinate into a unique value"""
#         return np.dot(coord.reshape(-1, self.dim), self.hash_vec)

#     def splat(self, x):
#         return self.S.dot(x)
    
#     def slice(self, y):
#         return self.S.T.dot(y)
    
#     def blur(self, x):
#         """Blur a bilateral-space vector with a 1 2 1 kernel in each dimension"""
#         assert x.shape[0] == self.nvertices
#         out = 2 * self.dim * x
#         for blur in self.blurs:
#             out = out + blur.dot(x)
#         return out

#     def filter(self, x):
#         """Apply bilateral filter to an input x"""
#         return self.slice(self.blur(self.splat(x))) /  \
#                self.slice(self.blur(self.splat(np.ones_like(x))))
    
    


# def bistochastize(grid, maxiter=10):
#     """Compute diagonal matrices to bistochastize a bilateral grid"""
#     m = grid.splat(np.ones(grid.npixels))
#     n = np.ones(grid.nvertices)
#     for i in range(maxiter):
#         n = np.sqrt(n * m / grid.blur(n))
#     # Correct m to satisfy the assumption of bistochastization regardless
#     # of how many iterations have been run.
#     m = n * grid.blur(n)
#     Dm = diags(m, 0)
#     Dn = diags(n, 0)
#     return Dn, Dm

# class BilateralSolver(object):
#     def __init__(self, grid, params):
#         self.grid = grid
#         self.params = params
#         self.Dn, self.Dm = bistochastize(grid)
    
#     def solve(self, x, w):
#         # Check that w is a vector or a nx1 matrix
#         if w.ndim == 2:
#             assert(w.shape[1] == 1)
#         elif w.dim == 1:
#             w = w.reshape(w.shape[0], 1)
#         A_smooth = (self.Dm - self.Dn.dot(self.grid.blur(self.Dn)))
#         w_splat = self.grid.splat(w)
#         A_data = diags(w_splat[:,0], 0)
#         A = self.params["lam"] * A_smooth + A_data
#         xw = x * w
#         b = self.grid.splat(xw)
#         # Use simple Jacobi preconditioner
#         A_diag = np.maximum(A.diagonal(), self.params["A_diag_min"])
#         M = diags(1 / A_diag, 0)
#         # Flat initialization
#         y0 = self.grid.splat(xw) / w_splat
#         yhat = np.empty_like(y0)
#         for d in range(x.shape[-1]):
#             yhat[..., d], info = cg(A, b[..., d], x0=y0[..., d], M=M, maxiter=self.params["cg_maxiter"], tol=self.params["cg_tol"])
#         xhat = self.grid.slice(yhat)
#         return xhat
    
# def bilateral_solver_output(input_images, target, sigma_spatial = 24, sigma_luma = 4, sigma_chroma = 4) : 
    
#     # reference = np.array(Image.open(img_pth).convert('RGB'))
#     h, w = target.shape
#     confidence = np.ones((h, w)) * 0.999
    
#     grid_params = {
#         'sigma_luma' : sigma_luma, # Brightness bandwidth
#         'sigma_chroma': sigma_chroma, # Color bandwidth
#         'sigma_spatial': sigma_spatial # Spatial bandwidth
#     }

#     bs_params = {
#         'lam': 256, # The strength of the smoothness parameter
#         'A_diag_min': 1e-5, # Clamp the diagonal of the A diagonal in the Jacobi preconditioner.
#         'cg_tol': 1e-5, # The tolerance on the convergence in PCG
#         'cg_maxiter': 25 # The number of PCG iterations
#     }

#     grid = BilateralGrid(input_images.squeeze(0).permute(1,2,0).cpu(), **grid_params)

#     t = target.reshape(-1, 1).astype(np.double) 
#     c = confidence.reshape(-1, 1).astype(np.double) 
    
#     ## output solver, which is a soft value
#     output_solver = BilateralSolver(grid, bs_params).solve(t, c).reshape((h, w))

#     binary_solver = ndimage.binary_fill_holes(output_solver>0.5)
#     labeled, nr_objects = ndimage.label(binary_solver) 

#     nb_pixel = [np.sum(labeled == i) for i in range(nr_objects + 1)]
#     pixel_order = np.argsort(nb_pixel)
#     try : 
#         binary_solver = labeled == pixel_order[-2]
#     except: 
#         binary_solver = np.ones((h, w), dtype=bool)
    
#     return output_solver, binary_solver
    
    