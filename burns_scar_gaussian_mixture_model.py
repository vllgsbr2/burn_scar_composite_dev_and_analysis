from sklearn import mixture
import numpy as np
import matplotlib.pyplot as plt

# Number of samples per component
n_samples = 500

# Generate random sample, two components
np.random.seed(0)
C = np.array([[0.0, -0.1], [1.7, 0.4]])
X = np.r_[
    np.dot(np.random.randn(n_samples, 2), C),
    0.7 * np.random.randn(n_samples, 2) + np.array([-6, 3]),
]

lowest_bic = np.infty
bic = []
n_components_range = range(1, 7)
cv_types = ["spherical", "tied", "diag", "full"]
for cv_type in cv_types:
    for n_components in n_components_range:
        # Fit a Gaussian mixture with EM
        gmm = mixture.GaussianMixture(
            n_components=n_components, covariance_type=cv_type
        )
        gmm.fit(X)
        bic.append(gmm.bic(X))
        if bic[-1] < lowest_bic:
            lowest_bic = bic[-1]
            best_gmm = gmm



################################################################################
# https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.
# html#sklearn.mixture.GaussianMixture

# class sklearn.mixture.GaussianMixture(n_components=1, *, covariance_type='full',
# tol=0.001, reg_covar=1e-06, max_iter=100, n_init=1, init_params='kmeans',
# weights_init=None, means_init=None, precisions_init=None, random_state=None,
# warm_start=False, verbose=0, verbose_interval=10)

burn_scar_2250nm_data = ''
burn_scar_0865nm_data = ''

X = #2D array 1 to 1 pixels of 2.25(row1) and 0.86(row2), cols are pixel BRF
n_components = 2 #(just 2 components)
train_X =
test_X  =
gmm = mixture.GaussianMixture(n_components=n_components, *, covariance_type='full',
tol=0.001, reg_covar=1e-06, max_iter=100, n_init=1, init_params='kmeans',
weights_init=None, means_init=None, precisions_init=None, random_state=None,
warm_start=False, verbose=0, verbose_interval=10).fit(train_X)

gmm.predict(test_X)
