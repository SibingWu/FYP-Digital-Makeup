import numpy as np
import scipy.stats
import logging

log = logging.getLogger('colour_transfer')


def colour_transfer_by_pdf_transfer(original_image, target_image, num_iterations=15):
    '''
    original_image and target_image should be with same size
    '''
    n_dims = original_image.shape[1]
    
    if n_dims != 3:
        raise Exception('Original image must have 3 channels.')
    
    assert original_image.shape[1] == 3
    
    # Initialization: xj = (rj , gj , bj ) 
    # where rj , gj , bj are the red, green and blue components of pixel number j.
    # k ← 0 , x(0) ← x
    xk = original_image.T
    y = target_image.T
    
    
    # Repeat
    for i in range(num_iterations):
        # take a rotation matrix R 
        # taking random rotations is sufficient to converge
        R = scipy.stats.special_ortho_group.rvs(dim=n_dims).astype(np.float32)  # Return a random rotation matrix
        
        # rotate the samples: xr ← Rx(k) and yr ← Ry
        xr = np.dot(R, xk)
        yr = np.dot(R, y)
        x_r = np.empty_like(xk)  # for storing remapped samples
        
        # project the samples on all axis i to get the marginals fi and gi
        for j in range(n_dims):
            low = min(xr[j].min(), yr[j].min())
            high = max(xr[j].max(), yr[j].max())
            
            # sample from distribution
            fi, bin_edges = np.histogram(xr[j], bins=300, range=[low, high])  # Compute the histogram of a dataset.
            gi, _ = np.histogram(yr[j], bins=300, range=[low, high])
            
            # find the 1D transformation ti that matches the marginals fi into gi
            # t(x) = Cy^-1(Cx(x)), x -> fi
            cx = fi.cumsum().astype(np.float32) # Return the cumulative sum of the elements along a given axis.
            cx /= cx[-1]  # cx[-1] is the total count
            
            cy = gi.cumsum().astype(np.float32)
            cy /= cy[-1]
            
            # Cy: xp=bin_edges[1:], fp=cy
            # Cy^-1: xp=cy, xp=bin_edges[1:]
            # t = Cy^-1 Cx
            t = np.interp(x=cx, xp=cy, fp=bin_edges[1:])  # Returns the one-dimensional piecewise linear interpolant to a function with given discrete data points (xp, fp), evaluated at x. f(xp) = fp, want to find y = f(x)
            
            # remap the samples xr according to the 1D transformations
            x_r[j] = np.interp(x=xr[j], xp=bin_edges[1:], fp=t, left=0, right=300)
            
        # rotate back the samples: x(k+1) ← R^−1xr
        # R x(k+1) = xr, want to find x(k+1)
        xk = np.linalg.solve(R, x_r)  # Computes the “exact” solution, x, of the linear matrix equation ax = b.
        log.info('Iteration %d/%d completed.', i + 1, num_iterations)
        
    return xk.T
        
