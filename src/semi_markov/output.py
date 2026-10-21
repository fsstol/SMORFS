from __future__ import annotations

import numpy as np


# The result of evaluating a real function at a vector of times, and optionally its gradient and Hessian matrices
class EvalResult:
    order: int
    params: int

    value: np.ndarray
    gradient: [None, np.ndarray]
    hessian: [None, np.ndarray]

    def __init__(self, value: np.ndarray, gradient: np.ndarray, hessian: np.ndarray):
        self.value = value  # shape (n_t)
        self.gradient = None  # shape (p, n_t)
        self.hessian = None  # shape (p, p, n_t)

        self.order = 0
        self.params = 0
        if gradient is not None:
            self.order = 1
            self.params = gradient.shape[0]
            self.gradient = gradient  # shape (p, n_t)
        if hessian is not None:
            self.order = 2
            self.hessian = hessian  # shape (p, p, n_t)

    def __repr__(self) -> str:
        n_t = self.value.shape[0]
        s = f"EvalResult[order={self.order}, params={self.params}, times={n_t}]"
        return s
