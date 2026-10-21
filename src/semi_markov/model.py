from __future__ import annotations

import numpy as np

from .risk import *
from .output import *


class Model(ABC):
    options: FourierOptions
    name: str
    risk_indices: dict[Risk, int]  # ordered dict of starting indices for each parameter set
    eval_order: int

    # To be filled automatically when updating the model
    n_params: int

    def __repr__(self):
        return self.name

    def print_parameters(self):
        print(f"{self.name} parameters:")
        for risk in self.risk_indices.keys():
            print("\t", risk, risk.get_params(), "->", risk.get_natural_params())

    def __init__(self, name: str):
        self.options = FourierOptions()
        self.name = name
        self.risk_indices = dict()
        self.n_params = 0
        self.eval_order = 0

        self.newton_time = 0.0

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def get_prevalence(self, t: np.ndarray, t0) -> dict[State, EvalResult]:
        pass

    @abstractmethod
    def get_arrival(self, t: np.ndarray, t0, first_states: list = None) -> dict[State, EvalResult]:
        pass

    @abstractmethod
    def set_parameters(self, params: np.ndarray):
        pass

    @abstractmethod
    def get_parameters(self) -> np.ndarray:
        pass

    def parameter_step(self, delta: np.ndarray):
        assert len(delta.shape) == 1
        assert delta.shape[0] == self.n_params
        self.set_parameters(self.get_parameters() + delta)

    # Newton step -H^+ g, or with saddle_free -|H|^+ g; returns the eigenvalues and eigenvectors of H (C++)
    def _newton_delta(self, hessian, gradient, saddle_free: bool):
        p = self.n_params
        hessian, gradient = as_c(hessian), as_c(gradient)
        delta, eig, P = np.empty(p), np.empty(p), np.empty((p, p))
        if lib.newton_step(c_ptr(hessian), c_ptr(gradient), p, int(saddle_free), c_ptr(delta), c_ptr(eig), c_ptr(P)) != 0:
            raise np.linalg.LinAlgError("Hessian not finite, or its eigen-decomposition failed")
        return delta, eig, P

    def _print_step(self, eig, P, delta, covariance=False):
        print("eigenvalues:")
        print(eig)
        print("eigenvectors:")
        print(np.round(P, 3))
        if covariance:
            cov = P @ np.diag(np.where(np.abs(eig) > len(eig) * np.finfo(float).eps * np.abs(eig).max(), 1.0 / eig, 0.0)) @ P.T
            print("covariance:")
            print(cov)
            print("sigmas:")
            print(np.sqrt(np.diag(cov)))
        print("delta:", delta)

    def newton_raphson_step(self, hessian, gradient):
        assert gradient.shape == (self.n_params,)
        assert hessian.shape == (self.n_params, self.n_params)

        t0 = time.perf_counter()
        delta, eig, P = self._newton_delta(hessian, gradient, saddle_free=False)
        if self.options.verbose:
            self._print_step(eig, P, delta)
        self.parameter_step(delta)
        t1 = time.perf_counter()
        self.newton_time += t1 - t0

        return eig

    def saddle_free_newton_step(self, hessian, gradient):
        assert gradient.shape == (self.n_params,)
        assert hessian.shape == (self.n_params, self.n_params)

        t0 = time.perf_counter()
        delta, eig, P = self._newton_delta(hessian, gradient, saddle_free=True)
        if self.options.verbose:
            self._print_step(eig, P, delta, covariance=True)
        self.parameter_step(delta)
        t1 = time.perf_counter()

        self.newton_time += t1 - t0
        return eig

    def gauss_newton_step(self, residuals: np.ndarray, jacobian: np.ndarray):
        # residuals: current vector of residual values, size r
        # jacobian: current jacobian matrix of residual functions, shape (r, p)
        assert len(residuals.shape) == 1
        assert len(jacobian.shape) == 2
        assert residuals.shape[0] == jacobian.shape[0]
        assert jacobian.shape[1] == self.n_params

        residuals, jacobian = as_c(residuals), as_c(jacobian)
        delta = np.empty(self.n_params)
        if lib.least_squares(c_ptr(jacobian), c_ptr(residuals), jacobian.shape[0], self.n_params, c_ptr(delta)) != 0:
            raise np.linalg.LinAlgError("Jacobian not finite")
        self.parameter_step(delta)

    def newton_flow_step(self, hessian, gradient, t: float):
        assert gradient.shape == (self.n_params,)
        assert hessian.shape == (self.n_params, self.n_params)

        t0 = time.perf_counter()
        p = self.n_params
        hessian, gradient, params = as_c(hessian), as_c(gradient), as_c(self.get_parameters())
        new_params, eig, P = np.empty(p), np.empty(p), np.empty((p, p))
        if lib.newton_flow_step(c_ptr(hessian), c_ptr(gradient), c_ptr(params), t, p, c_ptr(new_params), c_ptr(eig), c_ptr(P)) != 0:
            raise np.linalg.LinAlgError("Hessian not finite, or its eigen-decomposition failed")
        if self.options.verbose:
            self._print_step(eig, P, new_params - params)
        self.set_parameters(new_params)
        t1 = time.perf_counter()

        self.newton_time += t1 - t0
        return eig

    def langevin_step_euler(self, mu0, sigma0, grad0, hess0, step: float, Id):
        assert grad0.shape == (self.n_params,)
        assert hess0.shape == (self.n_params, self.n_params)

        t0 = time.perf_counter()
        p = self.n_params
        args = [as_c(x) for x in (mu0, sigma0, grad0, hess0)]
        mu_t, sigma_t = np.empty(p), np.empty((p, p))
        lib.langevin_step_euler(*[c_ptr(x) for x in args], step, c_ptr(as_c(Id)), p, c_ptr(mu_t), c_ptr(sigma_t))
        self.set_parameters(mu_t)
        t1 = time.perf_counter()

        self.newton_time += t1 - t0
        return mu_t, sigma_t

    # Langevin step (C++); if hess0 cannot be decomposed, sigma0 is returned and the parameters are not moved
    def langevin_step(self, sigma0, grad0, hess0, step: float):
        assert grad0.shape == (self.n_params,)
        assert hess0.shape == (self.n_params, self.n_params)

        t0 = time.perf_counter()
        p = self.n_params
        sigma0_c, grad0_c, hess0_c = as_c(sigma0), as_c(grad0), as_c(hess0)
        delta, sigma_t = np.empty(p), np.empty((p, p))
        if lib.langevin_step(c_ptr(sigma0_c), c_ptr(grad0_c), c_ptr(hess0_c), step, p, c_ptr(delta), c_ptr(sigma_t)) != 0:
            return sigma0
        self.parameter_step(delta)

        t1 = time.perf_counter()
        self.newton_time += t1 - t0
        return sigma_t

class State:
    name: str
    transitions: dict[State, Risk]  # connected states and their associated risk object

    def __init__(self, name):
        self.name = name
        self.transitions = dict()
        self.ptr = lib.create_state()
        self.terminal = False

    @property
    def terminal(self) -> bool:
        return self._terminal

    @terminal.setter
    def terminal(self, value: bool):  # kept in the C++ state
        self._terminal = bool(value)
        lib.set_terminal(self.ptr, int(self._terminal))

    def __del__(self):
        if lib is not None:
            lib.destroy_state(self.ptr)

    # add transition with risk to state other
    def add_transition(self, other: State, risk: Risk):
        if lib.add_transition(self.ptr, other.ptr, risk.ptr) != 0:
            raise ValueError(f"{self.name} already has a transition to {other.name}")
        self.transitions[other] = risk

    def __repr__(self):
        return repr(self.name)


LIFETIME_STATE = State("Lifetime")
SURVIVAL_STATE = State("Survival")
SPECIAL_STATES = [SURVIVAL_STATE, LIFETIME_STATE]
