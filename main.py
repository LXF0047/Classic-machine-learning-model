from sklearn.datasets import load_iris
import pandas as pd
import svm_model


class ClassicMachineLearningModels(object):
    def __init__(self):
        self.svc = svm_model.svc()
