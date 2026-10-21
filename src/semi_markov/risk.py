from abc import ABC, abstractmethod

import numpy as np

from .base import *


# A risk (transition hazard) is a handle to a C++ risk, which holds its parameters and evaluates it
class Risk(ABC):
    name: str
    ptr: ptr_struct = None

    def __repr__(self):
        return repr(self.name)

    @abstractmethod
    # get number of parameters
    def get_n_params(self) -> int:
        pass

    # get parameter values
    def get_params(self) -> np.ndarray:
        values = np.empty(self.get_n_params(), dtype=float)
        lib.get_params(self.ptr, values.ctypes.data_as(ptr_double))
        return values

    @abstractmethod
    # parameters on their natural scale (the log-scale ones exponentiated)
    def get_natural_params(self) -> np.ndarray:
        pass

    # set parameter values
    def set_params(self, values: np.ndarray):
        values = np.ascontiguousarray(values, dtype=float)
        lib.set_params(self.ptr, values.ctypes.data_as(ptr_double))


class ExponentialStable(Risk):
    # parameter: log rate

    def __init__(self, name, rate=1.0):
        self.name = name
        self.ptr = lib.create_exponential(rate)

    def __del__(self):
        lib.destroy_exponential(self.ptr)

    def get_n_params(self) -> int:
        return 1

    log_rate = property(lambda self: self.get_params()[0])
    rate = property(lambda self: np.exp(self.log_rate))

    def get_natural_params(self) -> np.ndarray:
        return np.exp(self.get_params())


class WeibullStable(Risk):
    # parameters: log shape and log scale

    def __init__(self, name, k=2.0, s=1.0):
        self.name = name
        self.ptr = lib.create_weibull(k, s)

    def __del__(self):
        lib.destroy_weibull(self.ptr)

    def get_n_params(self) -> int:
        return 2

    lk = property(lambda self: self.get_params()[0])
    ls = property(lambda self: self.get_params()[1])
    k = property(lambda self: np.exp(self.lk))
    s = property(lambda self: np.exp(self.ls))

    def get_natural_params(self) -> np.ndarray:
        return np.exp(self.get_params())


class GompertzStable(Risk):
    # h(t) = rate exp(shape t); parameters: shape (any sign) and log rate

    def __init__(self, name, shape=0.1, rate=1e-3):
        self.name = name
        self.ptr = lib.create_gompertz(shape, rate)

    def __del__(self):
        lib.destroy_risk(self.ptr)

    def get_n_params(self) -> int:
        return 2

    shape = property(lambda self: self.get_params()[0])
    rate = property(lambda self: np.exp(self.get_params()[1]))

    def get_natural_params(self) -> np.ndarray:
        return np.array([self.shape, self.rate])


class LogNormalStable(Risk):
    # log T ~ normal(meanlog, sdlog); parameters: meanlog and log sdlog

    def __init__(self, name, meanlog=0.0, sdlog=1.0):
        self.name = name
        self.ptr = lib.create_lognormal(meanlog, sdlog)

    def __del__(self):
        lib.destroy_risk(self.ptr)

    def get_n_params(self) -> int:
        return 2

    meanlog = property(lambda self: self.get_params()[0])
    sdlog = property(lambda self: np.exp(self.get_params()[1]))

    def get_natural_params(self) -> np.ndarray:
        return np.array([self.meanlog, self.sdlog])


class LogLogisticStable(Risk):
    # S(t) = 1 / (1 + (t / scale)^shape); parameters: log shape and log scale

    def __init__(self, name, shape=1.0, scale=1.0):
        self.name = name
        self.ptr = lib.create_loglogistic(shape, scale)

    def __del__(self):
        lib.destroy_risk(self.ptr)

    def get_n_params(self) -> int:
        return 2

    shape = property(lambda self: np.exp(self.get_params()[0]))
    scale = property(lambda self: np.exp(self.get_params()[1]))

    def get_natural_params(self) -> np.ndarray:
        return np.exp(self.get_params())
