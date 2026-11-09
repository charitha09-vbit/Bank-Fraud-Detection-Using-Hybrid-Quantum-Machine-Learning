
import os
import numpy as np
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import resample
from sklearn.ensemble import HistGradientBoostingClassifier

warnings.filterwarnings("ignore")

# ─────────────────────────────
# DATA GENERATION
# ─────────────────────────────
def generate_data(n_samples=6000):
    rng = np.random.default_rng(42)

    amount = rng.exponential(scale=350, size=n_samples).clip(1, 15000)
    time = rng.uniform(0, 24, size=n_samples)
    device = rng.integers(0, 2, size=n_samples)
    frequency = rng.integers(1, 50, size=n_samples)
    transactions = rng.integers(1, 300, size=n_samples)
    location = rng.integers(0, 2, size=n_samples)
    ip_risk = rng.beta(1.5, 6, size=n_samples)
    txn_type = rng.integers(0, 2, size=n_samples)

    avg_amount = np.mean(amount)
    amount_dev = amount / (avg_amount + 1)

    X = np.column_stack([
        amount, time, device, frequency,
        transactions, location, ip_risk, txn_type, amount_dev
    ])

    y = create_labels(X)

    X += rng.normal(0, X.std(axis=0) * 0.1, size=X.shape)

    flip = rng.choice(n_samples, int(n_samples * 0.02), replace=False)
    y[flip] = 1 - y[flip]

    return X, y


def create_labels(X):
    amount = X[:, 0]
    time = X[:, 1]
    device = X[:, 2]
    frequency = X[:, 3]
    transactions = X[:, 4]
    location = X[:, 5]
    ip_risk = X[:, 6]
    txn_type = X[:, 7]
    amount_dev = X[:, 8]

    score = np.zeros(len(X))

    score += ((amount > 3000) & ((time < 5) | (time > 23))) * 2
    score += (ip_risk > 0.75) * 1.5
    score += ((device == 1) & (location == 1)) * 2.5
    score += (frequency > 35) * 2

    score += (np.cos(time * np.pi / 12) > 0.8) * (ip_risk > 0.5) * 3
    score += (amount_dev > 4) * (frequency > 20) * 2.5
    score += ((device == 1) & (txn_type == 1) & (location == 1)) * 3
    score += ((amount < 50) & (transactions > 200)) * 2

    dist = (amount_dev / 10)**2 + (ip_risk)**2
    score += (dist > 0.5) * (location == 1) * 5

    return (score > 5).astype(int)


# ─────────────────────────────
# PREPROCESS
# ─────────────────────────────
def preprocess(X, y):
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42), scaler


# ─────────────────────────────
# BALANCE
# ─────────────────────────────
def oversample(X, y):
    X0 = X[y == 0]
    X1 = X[y == 1]

    y0 = y[y == 0]
    y1 = y[y == 1]

    X1_up, y1_up = resample(X1, y1, replace=True, n_samples=len(X0), random_state=42)

    return np.vstack([X0, X1_up]), np.hstack([y0, y1_up])


# ─────────────────────────────
# MODELS (FIXED)
# ─────────────────────────────
def train_classical(X, y):
    X, y = oversample(X, y)

    model = HistGradientBoostingClassifier(
        max_iter=200,
        max_depth=5,
        learning_rate=0.05,
        l2_regularization=1.0,
        random_state=42
    )
    model.fit(X, y)
    return model


def quantum_transform(X):
    return np.concatenate([np.sin(X), np.cos(X)], axis=1)


def train_hybrid(X, y):
    Xq = quantum_transform(X)
    Xq, y = oversample(Xq, y)

    model = HistGradientBoostingClassifier(
        max_iter=600,
        max_depth=10,
        learning_rate=0.05,
        random_state=42
    )
    model.fit(Xq, y)
    return model


# ─────────────────────────────
# PREDICTION (FIXED)
# ─────────────────────────────
def predict(model_a, model_b, scaler, row):
    row = scaler.transform(row)

    q = quantum_transform(row)

    pa = model_a.predict_proba(row)[0, 1]
    pb = model_b.predict_proba(q)[0, 1]

    # weighted ensemble (FIXED)
    final = 0.3 * pa + 0.7 * pb

    return pa, pb, final


def decision(p):
    if p >= 0.60:
        return "FRAUD"
    elif p < 0.35:
        return "NORMAL"
    else:
        return "SUSPICIOUS"


# ─────────────────────────────
# MAIN
# ─────────────────────────────
def main():
    X, y = generate_data()

    (X_train, X_test, y_train, y_test), scaler = preprocess(X, y)

    print("Training models...")

    model_a = train_classical(X_train, y_train)
    model_b = train_hybrid(X_train, y_train)

    acc_a = accuracy_score(y_test, model_a.predict(X_test))
    acc_b = accuracy_score(y_test, model_b.predict(quantum_transform(X_test)))

    print(f"Classical Accuracy: {acc_a:.3f}")
    print(f"Hybrid Accuracy:   {acc_b:.3f}")

    while True:
        print("\nEnter transaction:")

        vals = list(map(float, input().split()))
        row = np.array([vals + [vals[0] / (X[:,0].mean()+1)]])

        pa, pb, pf = predict(model_a, model_b, scaler, row)

        print("\nClassical:", decision(pa), pa)
        print("Hybrid:   ", decision(pb), pb)
        print("Final:    ", decision(pf), pf)


if __name__ == "__main__":
    main()
```
