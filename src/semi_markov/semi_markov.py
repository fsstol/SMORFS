from __future__ import annotations

import numpy as np

from .base import *
from .model import *

# Upper limit of the quadrature of general states' transforms
MAX_TIME = 2e2


class SemiMarkov(Model):
    state_indices: dict[State, int]  # ordered dict of indices for each state
    general_indices: dict[State, int]  # ordered dict of indices for each general state
    exponential_indices: dict[State, int]  # ordered dict of indices for each general state
    initial: State | None

    # To be filled automatically when updating the model
    n_states: int
    n_general: int
    n_exponential: int

    def __init__(self, name: str):
        super().__init__(name)
        self.state_indices = dict()
        self.general_indices = dict()
        self.exponential_indices = dict()
        self.n_states = 0
        self.n_general = 0
        self.n_exponential = 0
        self.initial = None

        # For profiling
        self.quad_time = 0.0
        self.dense_time = 0.0
        self.stack_time = 0.0
        self.general_time = 0.0
        self.exponential_time = 0.0
        self.concat_time = 0.0
        self.unstack_time = 0.0
        self.invert_time = 0.0
        self.rhs_time = 0.0
        self.prevalence_time = 0.0
        self.arrival_time = 0.0
        self.transforms_time = 0.0
        self.series_time = 0.0

        self.ptr = lib.create_semi_markov()

    def __del__(self):
        lib.destroy_semi_markov(self.ptr)

    def add_state(self, state: State):
        assert state not in SPECIAL_STATES
        self.state_indices[state] = -1
        lib.add_state(self.ptr, state.ptr)

    def set_initial(self, state: State):
        assert state in self.state_indices.keys()
        self.initial = state
        lib.set_initial(self.ptr, state.ptr)

    # The layout (state order, general states first, and parameter offsets) is computed by the C++ model
    def update(self):
        if lib.update(self.ptr) != 0:
            raise ValueError(f"{self.name}: a transition leads to a state not in the model")
        sizes = np.empty(4, dtype=np.int64)
        lib.get_sizes(self.ptr, sizes.ctypes.data_as(ptr_int))
        self.n_states, self.n_general, n_risks, self.n_params = (int(x) for x in sizes)
        self.n_exponential = self.n_states - self.n_general
        state_ptrs = (ctypes.c_void_p * self.n_states)()
        risk_ptrs = (ctypes.c_void_p * n_risks)()
        offsets = np.empty(n_risks, dtype=np.int64)
        lib.get_layout(self.ptr, state_ptrs, risk_ptrs, offsets.ctypes.data_as(ptr_int))

        address = lambda obj: ctypes.cast(obj.ptr, ctypes.c_void_p).value
        states = {address(s): s for s in self.state_indices}
        risks = {address(r): r for s in states.values() for r in s.transitions.values()}
        order = [states[a] for a in state_ptrs]
        self.state_indices = {s: i for i, s in enumerate(order)}
        self.general_indices = {s: i for i, s in enumerate(order[:self.n_general])}
        self.exponential_indices = {s: i for i, s in enumerate(order[self.n_general:])}
        self.risk_indices = {risks[a]: int(o) for a, o in zip(risk_ptrs, offsets)}

    def set_parameters(self, params: np.ndarray):
        assert len(params.shape) == 1
        assert params.shape[0] == self.n_params, f"Incorrect number of parameters (expected {self.n_params}, got {params.shape[0]})"
        params = np.ascontiguousarray(params, dtype=float)
        lib.sm_set_parameters(self.ptr, params.ctypes.data_as(ptr_double))

    def get_parameters(self):
        params = np.empty(self.n_params, dtype=float)
        lib.sm_get_parameters(self.ptr, params.ctypes.data_as(ptr_double))
        return params

    def __get_height(self):
        p = self.n_params
        if self.eval_order == 0:
            h = 1
        elif self.eval_order == 1:
            h = 1 + p
        else:
            h = 1 + p + p * p
        return h

    # Views of the C++ output (states, lifetime, survival) x (value, gradient, Hessian) x times
    def __get_eval_results(self, stack, n_t):
        n = self.n_states
        p = self.n_params
        vals = stack[:, 0, :]
        gradients = stack[:, 1:1 + p, :] if self.eval_order >= 1 else [None] * (n+2)
        hessians = stack[:, 1 + p:, :].reshape((n + 2, p, p, n_t)) if self.eval_order >= 2 else [None] * (n+2)
        keys = list(self.state_indices) + [LIFETIME_STATE, SURVIVAL_STATE]
        return {key: EvalResult(vals[i], gradients[i], hessians[i]) for i, key in enumerate(keys)}

    # Transforms at every frequency and the Fourier series at times t (C++)
    def __get_time_eval(self, prevalence: bool, t: np.ndarray, t0, cutoff_states=(), cumulative=False) -> dict[State, EvalResult]:
        assert self.initial is not None, "Must set initial state"
        assert len(t.shape) == 1
        n = self.n_states
        n_t = t.shape[0]
        states = list(self.state_indices.keys())
        if isinstance(cumulative, list):
            cum = [s in cumulative for s in states] + [False]
        elif cumulative == True:
            cum = [True] * (n+1)
        else:
            cum = [False] * (n+1)
        flags = [np.array(x, dtype=np.int64) for x in (cum, [s in cutoff_states for s in states])]
        t = np.ascontiguousarray(t, dtype=float)
        stack = np.empty((n+2, self.__get_height(), n_t), dtype=float)
        timings = np.zeros(5, dtype=float)
        o = self.options
        status = lib.evaluate(self.ptr, int(prevalence), *[x.ctypes.data_as(ptr_int) for x in flags], t.ctypes.data_as(ptr_double), n_t,
                              t0, o.A, o.l, o.periods, MAX_TIME, self.eval_order, stack.ctypes.data_as(ptr_double),
                              timings.ctypes.data_as(ptr_double))
        if status == 2:
            raise RuntimeError(f"{self.name}: a state or transition was added after update(); call update() again")
        if status != 0:
            raise np.linalg.LinAlgError("Singular matrix")

        self.quad_time += timings[0]
        self.general_time += timings[0]
        self.exponential_time += timings[1]
        self.invert_time += timings[2] + timings[3]
        self.transforms_time += timings[:4].sum()
        self.series_time += timings[4]
        return self.__get_eval_results(stack, n_t)

    def get_prevalence(self, t: np.ndarray, t0, prefix: str = None) -> dict[State, EvalResult]:
        t1 = time.perf_counter()
        res = self.__get_time_eval(True, t, t0)
        t2 = time.perf_counter()
        self.prevalence_time += t2-t1
        return res

    def get_arrival(self, t: np.ndarray, t0, cutoff_states: list = None, prefix: str = None, cumulative=False) -> dict[State, EvalResult]:
        t1 = time.perf_counter()
        res = self.__get_time_eval(False, t, t0, cutoff_states or (), cumulative)
        t2 = time.perf_counter()
        self.arrival_time += t2-t1
        return res

    def print_times(self, n_digits=3):
        print("\ttransforms:", round(self.transforms_time, n_digits))
        print("\t\tgeneral:", round(self.general_time, n_digits))
        print("\t\t\tquad :", round(self.quad_time, n_digits))
        print("\t\t\tdense:", round(self.dense_time, n_digits))
        print("\t\t\tstack:", round(self.stack_time, n_digits))
        print("\t\texponential:", round(self.exponential_time, n_digits))
        print("\t\tconcat:", round(self.concat_time, n_digits))
        print("\t\tunstack:", round(self.unstack_time, n_digits))
        print("\t\tinvert:", round(self.invert_time, n_digits))
        print("\t\t\trhs:", round(self.rhs_time, n_digits))
        print("\tseries:", round(self.series_time, n_digits))
        print("\tnewton:", round(self.newton_time, n_digits))
