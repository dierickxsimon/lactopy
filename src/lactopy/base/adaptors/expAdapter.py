import copy
import numpy as np
from scipy.optimize import curve_fit


class ExpAdaptor:
    @property
    def params(self):
        return {
            "a": self.a,
            "b": self.b,
            "c": self.c,
            "min_domain": self.min_domain,
            "max_domain": self.max_domain,
        }

    def fit(self, X, Y):
        def exp_func(x, a, b, c):
            return a * np.exp(b * x) + c

        popt, _ = curve_fit(exp_func, X, Y, maxfev=10000)
        self.a, self.b, self.c = popt
        self.min_domain = X.min()
        self.max_domain = X.max()
        return self

    def predict(self, X):
        return self.a * np.exp(self.b * X) + self.c

    def predict_inverse(self, Y):
        if np.any(Y - self.c <= 0):
            raise ValueError("No real solution for given Y")
        X = np.log((Y - self.c) / self.a) / self.b
        if np.any((X < self.min_domain) | (X > self.max_domain)):
            raise ValueError("Solution out of domain")
        return X

    def dxdt(self):
        r_obj = copy.deepcopy(self)

        def deriv(X):
            return r_obj.a * r_obj.b * np.exp(r_obj.b * X)

        r_obj.deriv = deriv
        return r_obj
