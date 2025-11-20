import torch
from sklearn.decomposition import PCA
from scipy.optimize import curve_fit
import numpy as np

def f(n, alpha, c, b):
    n = np.asarray(n, dtype=float)
    return c / (n ** alpha) + b


def fit_power_law(sorted_list_var_ratio, bounds=(10, 30)):
    arr = np.asarray(sorted_list_var_ratio, dtype=float)
    if arr.size == 0:
        return np.array([np.nan, np.nan, np.nan]), np.full((3, 3), np.nan)

    if bounds is not None:
        i0, i1 = bounds
        i0 = max(1, int(i0))  # ensure >=1 to avoid division by zero
        i1 = min(int(i1), arr.size)
        x = np.arange(i0, i1)  # same indexing as used elsewhere in the notebook
        y = arr[i0:i1]
    else:
        x = np.arange(1, arr.size + 1)
        y = arr

    if y.size < 3:
        return np.array([np.nan, np.nan, np.nan]), np.full((3, 3), np.nan)

    try:
        popt, pcov = curve_fit(f, x, y, p0=[0.5, 0.1, 0.0], maxfev=10000)
        residuals = y - f(x, *popt)
        msr = np.mean(residuals ** 2)
        return popt, pcov
    except Exception as e:
        print(f"fit_power_law: curve_fit failed: {e}")
        return np.array([np.nan, np.nan, np.nan]), np.full((3, 3), np.nan)


def fetch_spectrum_pca(activation, device=None, n_components=40):
    if torch.is_tensor(activation):
        X_t = activation.detach().cpu()
    else:
        X_t = torch.tensor(activation).cpu()
    X = X_t.view(X_t.size(0), -1).numpy()
    pca = PCA(n_components=min(n_components, X.shape[1], X.shape[0]))
    pca.fit(X)
    var_ratio = np.array(pca.explained_variance_ratio_, dtype=float)
    var = np.array(pca.explained_variance_, dtype=float)
    return var_ratio.tolist(), var.tolist()