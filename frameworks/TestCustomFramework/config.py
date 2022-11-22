from enum import Enum

class FCAlgos(Enum):
    FC_NEIGHBORS = 1
    FC_LDA = 2
    FC_PLS = 3

feature_construction_order_0 = [
    (FCAlgos.FC_NEIGHBORS, 1),
    (FCAlgos.FC_LDA, 1),
    (FCAlgos.FC_PLS, 1)
]

feature_construction_order_1 = [
    (FCAlgos.FC_NEIGHBORS, 1),
    (FCAlgos.FC_PLS, 1),
    (FCAlgos.FC_LDA, 1),
]

feature_construction_order_1_5_features = [
    (FCAlgos.FC_NEIGHBORS, 3),
    (FCAlgos.FC_PLS, 1),
    (FCAlgos.FC_LDA, 1),
]

feature_construction_order_2 = [
    (FCAlgos.FC_LDA, 1),
    (FCAlgos.FC_NEIGHBORS, 1),
    (FCAlgos.FC_PLS, 1)
]

feature_construction_order_3 = [
    (FCAlgos.FC_PLS, 1),
    (FCAlgos.FC_NEIGHBORS, 1),
    (FCAlgos.FC_LDA, 1),
]

feature_construction_order_4 = [
    (FCAlgos.FC_LDA, 1),
    (FCAlgos.FC_PLS, 1),
    (FCAlgos.FC_NEIGHBORS, 1),
]

feature_construction_order_5 = [
    (FCAlgos.FC_PLS, 1),
    (FCAlgos.FC_LDA, 1),
    (FCAlgos.FC_NEIGHBORS, 1),
]

feature_construction_order_5_5_features = [
    (FCAlgos.FC_PLS, 1),
    (FCAlgos.FC_LDA, 1),
    (FCAlgos.FC_NEIGHBORS, 3),
]
