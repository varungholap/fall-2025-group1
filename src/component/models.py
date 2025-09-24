import numpy as np

class LinUCB:
    def __init__(self, alpha=1.5, d=None):
        self.alpha = alpha
        self.d = d
        self.A = None
        self.b = None
        self.initialized = False

    def _maybe_init(self, d):
        if not self.initialized:
            self.d = d if self.d is None else self.d
            self.A = np.eye(self.d)
            self.b = np.zeros((self.d, 1))
            self.initialized = True

    def select(self, X: np.ndarray) -> int:
        K, D = X.shape
        self._maybe_init(D)
        A_inv = np.linalg.inv(self.A)
        theta = A_inv @ self.b
        scores = []
        for k in range(K):
            x = X[k].reshape(-1, 1)
            mu = float(theta.T @ x)
            sigma = float(np.sqrt(x.T @ A_inv @ x))
            scores.append(mu + self.alpha * sigma)
        return int(np.argmax(scores))

    def update(self, x: np.ndarray, reward: float):
        x = x.reshape(-1, 1)
        self.A += x @ x.T
        self.b += reward * x