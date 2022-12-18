import os
from unittest import TestCase
from ml_service.ml_service_flask import (
    GetModelsRest,
    GetHyperParamsRest
)

from werkzeug.exceptions import NotFound


class TestGetModelsRest(TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ['TEST'] = '1'

    def test_get(self):
        gmr = GetModelsRest()

        with self.assertRaises(NotFound):
            gmr.get('wrong_type')

        self.assertDictEqual(gmr.get('classification'), {
            'model_names': ['LogisticRegression', 'KNeighborsClassifier']
        })


class TestGetHyperParamsRest(TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ['TEST'] = '1'

    def test_get(self):
        ghpr = GetHyperParamsRest()

        self.assertIn('hyper_params', ghpr.get('LogisticRegression'))

        with self.assertRaises(NotFound):
            ghpr.get('LogisticRegressionWrong')

