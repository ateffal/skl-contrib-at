import pytest

from sklearn.utils.estimator_checks import check_estimator

from skl-contrib-at import TemplateEstimator
from skl-contrib-at import TemplateClassifier
from skl-contrib-at import TemplateTransformer


@pytest.mark.parametrize(
    "Estimator", [TemplateEstimator, TemplateTransformer, TemplateClassifier]
)
def test_all_estimators(Estimator):
    return check_estimator(Estimator)
