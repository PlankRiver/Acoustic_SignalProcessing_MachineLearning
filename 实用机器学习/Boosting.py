import numpy as np
import sklearn as sk

class GradientBoosting():
    def __init__(self,base_learner,n_learners,learning_rate):
        self.learners = [sk.clone(base_learner) for _ in range(n_learners)]
        self.learning_rate = learning_rate
    def fit(self,X,y):
        residual = y.copy()
        for learner in self.learners:
            learner.fit(X,y)
            residual -= self.learning_rate*learner.predict(X)
    def predict(self,X):
        preds = [learner.predict(X) for learner in self.learners]
        return np.array(preds).sum(axis=0)*self.learning_rate