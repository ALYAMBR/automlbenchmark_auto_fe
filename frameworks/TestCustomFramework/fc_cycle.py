from frameworks.TestCustomFramework.config import FCAlgos
from frameworks.TestCustomFramework.config import feature_construction_order_0
from frameworks.TestCustomFramework.config import feature_construction_order_1
from frameworks.TestCustomFramework.config import feature_construction_order_2
from frameworks.TestCustomFramework.config import feature_construction_order_3
from frameworks.TestCustomFramework.config import feature_construction_order_4
from frameworks.TestCustomFramework.config import feature_construction_order_5
import numpy as np
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cross_decomposition import PLSCanonical


def get_model_by_name(model_name, num_feats):
    if model_name == FCAlgos.FC_NEIGHBORS:
        return NeighborhoodComponentsAnalysis(n_components=num_feats)
    elif model_name == FCAlgos.FC_LDA:
        return LinearDiscriminantAnalysis(n_components=num_feats)
    elif model_name == FCAlgos.FC_PLS:
        return PLSCanonical(n_components=num_feats)


def construct_features(x, y=None, fitted_models=None, return_only_constructed=False):
    fitted_algos = [] if fitted_models is None else fitted_models
    is_first_iteration = True
    if y is None:
        for model in fitted_algos:
            print("Using model that has been traiend previously.")
            if is_first_iteration and return_only_constructed:
                x = model.transform(x)
                is_first_iteration = False
            else:
                x = np.concatenate((x, model.transform(x)), axis=1)
    else:
        for alg, num_feats in feature_construction_order_1:
            print("Using model that is training now.")
            model = get_model_by_name(alg, num_feats)
            model = model.fit(x, y)
            if is_first_iteration and return_only_constructed:
                x = model.transform(x)
                is_first_iteration = False
            else:
                x = np.concatenate((x, model.transform(x)), axis=1)
            fitted_algos.append(model)

    return x, fitted_algos