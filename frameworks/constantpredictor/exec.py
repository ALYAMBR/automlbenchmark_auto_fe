import logging

from sklearn.dummy import DummyClassifier, DummyRegressor

from automl.benchmark import TaskConfig
from automl.data import Dataset
from automl.results import save_predictions_to_file
from automl.utils import Timer


log = logging.getLogger(__name__)


def run(dataset: Dataset, config: TaskConfig):
    log.info("\n**** Constant predictor (sklearn dummy) ****\n")

    is_classification = config.type == 'classification'
    predictor = DummyClassifier(strategy='prior') if is_classification else DummyRegressor(strategy='median')

    encode = config.framework_params['encode'] if 'encode' in config.framework_params else False
    X_train = dataset.train.X_enc if encode else dataset.train.X
    y_train = dataset.train.y_enc if encode else dataset.train.y
    X_test = dataset.test.X_enc if encode else dataset.test.X
    y_test = dataset.test.y_enc if encode else dataset.test.y

    with Timer() as training:
        predictor.fit(X_train, y_train)
    predictions = predictor.predict(X_test)
    probabilities = predictor.predict_proba(X_test) if is_classification else None

    save_predictions_to_file(dataset=dataset,
                             output_file=config.output_predictions_file,
                             probabilities=probabilities,
                             predictions=predictions,
                             truth=y_test,
                             target_is_encoded=encode)

    return dict(
        models_count=1,
        training_duration=training.duration
    )
