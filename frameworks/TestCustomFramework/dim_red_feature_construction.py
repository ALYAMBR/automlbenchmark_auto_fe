from enum import Enum
import numpy as np
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cross_decomposition import PLSCanonical


class FCAlgos(Enum):
    FC_NEIGHBORS = 1
    FC_LDA = 2
    FC_PLS = 3


def get_model_by_name(model_name, num_feats):
    if model_name == FCAlgos.FC_NEIGHBORS:
        return NeighborhoodComponentsAnalysis(n_components=num_feats)
    elif model_name == FCAlgos.FC_LDA:
        return LinearDiscriminantAnalysis(n_components=num_feats)
    elif model_name == FCAlgos.FC_PLS:
        return PLSCanonical(n_components=num_feats)


def check_config(config):
    for model, num_features in config:
        if model == FCAlgos.FC_LDA or model == FCAlgos.FC_PLS:
            if num_features > 1:
                print("Config isn't correct. LDA and PLS can't"
                      " generate more than one feature "
                      "in binary classification task.")
                return False
    return True


class DimensionReductionFeatureConstruction:
    def __init__(self, config=[
            (FCAlgos.FC_PLS, 1),
            (FCAlgos.FC_LDA, 1),
            (FCAlgos.FC_NEIGHBORS, 3)],
            return_only_constructed=False,
            construct_independent=False) -> None:
        self._is_fitted = False
        self._fitted_algos = []
        self._config = config
        self._return_only_constructed = return_only_constructed
        self._construct_independent = construct_independent

    def fit(self, train_X, train_y):
        if check_config(self._config) is False:
            print("Fit hasn't executed.")
            return self
        is_first_iteration = True
        X = train_X.copy()
        for alg, num_features in self._config:
            print(f"Using {alg} algorithm that is training now.")
            model = get_model_by_name(alg, num_features)
            if self._construct_independent:
                model = model.fit(train_X, train_y)
            else:
                model = model.fit(X, train_y)
            if is_first_iteration and self._return_only_constructed:
                if self._construct_independent:
                    X = model.transform(train_X)
                else:
                    X = model.transform(X)
                is_first_iteration = False
            else:
                if self._construct_independent:
                    X = np.concatenate(
                        (X, model.transform(train_X)), axis=1)
                else:
                    X = np.concatenate(
                        (X, model.transform(X)), axis=1)
            self._fitted_algos.append(model)
        self._is_fitted = True
        return self

    def transform(self, full_X):
        if self._is_fitted is False:
            print("Transformer isn't fitted. Data returned without transformation.")
            return full_X
        is_first_iteration = True
        x = full_X.copy()
        for model in self._fitted_algos:
            if is_first_iteration and self._return_only_constructed:
                x = model.transform(x)
                is_first_iteration = False
            else:
                if self._construct_independent:
                    x = np.concatenate((x, model.transform(full_X)), axis=1)
                else:
                    x = np.concatenate((x, model.transform(x)), axis=1)
        return x

    def fit_transform(self, data_X, data_y):
        return self.fit(data_X, data_y).transform(data_X)

    def is_fitted(self):
        return self._is_fitted

    def get_fitted_algorithms(self):
        return self._fitted_algos
