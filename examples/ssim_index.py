import numpy as np

def SSIMIndex(X,Y, k1 = 0.01, k2 = 0.03):
    """
    Structural similarity index. Implementation based on the paper: Image Quality Assessment:
    From Error Visibility to Structural Similarity. k1 = 0.01, k2 = 0.03 are the
    default values used in the paper.
    """

    X = X.astype(float)
    Y = Y.astype(float)

    mu_x = X.mean()
    mu_y = Y.mean()

    sigma_xy = 1.0/(X.size - 1)*((X.ravel() - mu_x)*(Y.ravel() - mu_y)).sum()
    sigma_y = 1.0/(Y.size - 1)*((Y.ravel() - mu_y)** 2).sum()
    sigma_y = np.sqrt(sigma_y)
    sigma_x = 1.0/(X.size - 1)*((X.ravel() - mu_x)** 2).sum()
    sigma_x = np.sqrt(sigma_x)

    C1 = (k1*255)*(k1*255)
    C2 = (k2*255)*(k2*255)
    ssim = (2*mu_x*mu_y + C1)*(2*sigma_xy + C2)/((mu_x*mu_x + mu_y*mu_y + C1)\
           *(sigma_x*sigma_x + sigma_y*sigma_y + C2))
    return ssim