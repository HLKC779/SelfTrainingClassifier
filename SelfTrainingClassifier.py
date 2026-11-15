import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.semi_supervised import SelfTrainingClassifier

X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X / 16, y, test_size=300, stratify=y, random_state=0)

rng = np.random.default_rng(0)
labeled = []
for digit in range(10):
    idx = np.where(y_train == digit)[0]
    labeled.append(rng.choice(idx, 5, replace=False))
y_semi = np.full(y_train.shape, -1)
y_semi[np.concatenate(labeled)] = y_train[np.concatenate(labeled)]

model = LogisticRegression(C=10, max_iter=2000)
model.fit(X_train[np.concatenate(labeled)], y_train[np.concatenate(labeled)])
acc = model.score(X_test, y_test)
print(f"50 labels: {acc:.0%}")

self_training_model = SelfTrainingClassifier(
    model, criterion="threshold", threshold=0.5)
self_training_model.fit(X_train, y_semi)
acc = self_training_model.score(X_test, y_test)
print(f"Self-training with 10% confidence: {acc:.0%}")