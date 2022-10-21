from enum import Enum

class FCAlgos(Enum):
    FC_NEIGHBORS = 1
    FC_LDA = 2
    FC_PLS = 3

feature_construction_order = [
    (FCAlgos.FC_NEIGHBORS, 1),
    (FCAlgos.FC_LDA, 1),
    (FCAlgos.FC_PLS, 1)
]