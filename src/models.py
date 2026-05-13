from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score
import pandas as pd

class CrowdfundingTrainer:
    def __init__(self, model_type='xgb', params=None):
        self.model_type = model_type
        if params is None:
            params = self._get_default_params()
        self.model = self._init_model(params)

    def _get_default_params(self):
        if self.model_type == 'xgb':
            return {
                'eval_metric': 'logloss',
                'scale_pos_weight': 3,
                'max_depth': 16,
                'learning_rate': 0.1,
                'reg_alpha': 0.1,
                'reg_lambda': 0.25
            }
        return {}

    def _init_model(self, params):
        if self.model_type == 'xgb':
            return XGBClassifier(**params)
        elif self.model_type == 'rf':
            return RandomForestClassifier(**params)
        elif self.model_type == 'lr':
            return LogisticRegression(max_iter=1000, **params)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'auc': roc_auc_score(y_test, y_proba),
            'f1': f1_score(y_test, y_pred)
        }
        return metrics

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
