from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from deprecated import deprecated

@deprecated('Not used')
class XGBWrapper:
    def __init__(self, config:dict):
        self.model = XGBClassifier(max_depth=20, 
            verbosity=0, random_state=101)

@deprecated('Not used')
class GBRWrapper:
    def __init__(self, config:dict):
        self.model = GradientBoostingClassifier(n_estimators=20, 
            learning_rate=0.1, max_features=12, max_depth=20)