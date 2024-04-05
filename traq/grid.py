from collections import OrderedDict

from pyod.models.ecod import ECOD
from pyod.models.gmm import GMM
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF

# from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA


class Euclidean:
    def __init__(self):
        # This is equivalent to the multivariate Euclidean distance, scaled by
        # the standard deviation.
        self.obj = GMM(n_components=1, covariance_type="diag")

    def __getattr__(self, name):
        return getattr(self.obj, name)


class Mahalanobis:
    def __init__(self):
        # This is equivalent to the multivariate Mahalanobis distance (Euclidean
        # scaled by multivariate prevision).
        self.obj = GMM(n_components=1, covariance_type="full")

    def __getattr__(self, name):
        # Sometimes estimating the covariance can fail. Fall back to Euclidean
        # in this case.
        try:
            return getattr(self.obj, name)
        except:  # noqa: E722
            self.obj = GMM(n_components=1, covariance_type="diag")
            return getattr(self.obj, name)


cash = OrderedDict(
    {
        ECOD: {},
        IForest: {},
        KNN: {},  # slow
        LOF: {},  # slow
        # OCSVM: {},  # very slow
        PCA: {},
        HBOS: {},
        Euclidean: {},
        Mahalanobis: {},
    }
)
