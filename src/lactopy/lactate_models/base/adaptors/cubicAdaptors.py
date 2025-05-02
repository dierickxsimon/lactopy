import numpy as np
from scipy.interpolate import CubicSpline


class CubicAdaptor:
    def fit(self, X, Y):
        self._spline = CubicSpline(X, Y)
        self.min_domain = X.min()
        self.max_domain = X.max()
        return self

    def predict(self, X):
        return self._spline(X)

    def predict_inverse(self, Y):
        roots = self._spline.roots(Y)
        real_roots = [np.isreal(roots)].real

        real_roots_within_domain = real_roots[
            (real_roots >= self.min_domain) & (real_roots <= self.max_domain)
        ]
        if len(real_roots_within_domain) < 1:
            raise ("no solution was found")
        if len(real_roots_within_domain) > 1:
            raise ("to many roots where found")

        return real_roots_within_domain[0]

    def dxdt(self):
        self._spline = self._spline.derivative()
        return self
