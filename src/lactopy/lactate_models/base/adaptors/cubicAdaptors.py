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
        roots = np.array([])
        for i, coef in enumerate(self._spline.c.T):
            coeffs = coef.copy()
            coeffs[-1] -= Y
            poly_roots = np.roots(coeffs)
            real_roots = poly_roots[np.isreal(poly_roots)].real
            shifted_roots = real_roots + self._spline.x[i]
            valid_roots = shifted_roots[
                (shifted_roots >= self._spline.x[i])
                & (shifted_roots <= self._spline.x[i + 1])
            ]
            roots = np.append(roots, valid_roots)

        if len(roots) < 1:
            raise ValueError("no solution was found")
        if len(roots) > 2:
            raise ValueError("too many roots were found")

        return roots[0]

    def dxdt(self):
        self._spline = self._spline.derivative()
        return self
