from sklearn.model_selection import StratifiedKFold
import numpy as np
X = np.ones(10)
y = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
skf = StratifiedKFold(n_splits=3)
print(X)
print(y)
for train, test in skf.split(X, y):
    print("%s %s" % (train, test))
