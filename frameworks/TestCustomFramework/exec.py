import logging

import sklearn
from xgboost import XGBClassifier, XGBRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler

from amlb.benchmark import TaskConfig
from amlb.data import Dataset
from amlb.datautils import impute_array
from amlb.results import save_predictions
from amlb.utils import Timer, unsparsify
from frameworks.TestCustomFramework.fc_cycle import construct_features

log = logging.getLogger(__name__)


def run(dataset: Dataset, config: TaskConfig):
    log.info(f"\n**** Custom framework[sklearn v{sklearn.__version__}] ****\n")

    is_classification = config.type == 'classification'

    X_train, X_test = impute_array(*unsparsify(dataset.train.X_enc, dataset.test.X_enc, fmt='array'))
    y_train, y_test = unsparsify(dataset.train.y_enc, dataset.test.y_enc, fmt='array')

    # feature construction code
    # BASELINE
    # lda = LinearDiscriminantAnalysis(n_components=1)
    # lda = lda.fit(X_train, y_train)

    # log.info(X_train.shape)
    # X_train = np.concatenate((X_train, lda.transform(X_train)), axis=1)
    # X_test = np.concatenate((X_test, lda.transform(X_test)), axis=1)
    # log.info(X_train.shape)
    #
    #FC cycle
    X_train, fitted_fc_models = construct_features(X_train, y_train, None, return_only_constructed=False)
    print("Train:")
    print(X_train.shape)
    X_test, _ = construct_features(X_test, None, fitted_fc_models, return_only_constructed=False)
    print("Test:")
    print(X_test.shape)
    # --------------------------
    # Command to run: python runbenchmark.py TestCustomFramework ultrasmall 1h12c
    #

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    estimator = XGBClassifier if is_classification else XGBRegressor
    # estimator =  DecisionTreeClassifier if is_classification else DecisionTreeRegressor
    predictor = estimator(random_state=config.seed, **config.framework_params)

    with Timer() as training:
        predictor.fit(X_train, y_train)
    with Timer() as predict:
        predictions = predictor.predict(X_test)
    probabilities = predictor.predict_proba(X_test) if is_classification else None

    save_predictions(dataset=dataset,
                     output_file=config.output_predictions_file,
                     probabilities=probabilities,
                     predictions=predictions,
                     truth=y_test,
                     target_is_encoded=is_classification)

    return dict(
        models_count=1,
        training_duration=training.duration,
        predict_duration=predict.duration
    )
