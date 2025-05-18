import numpy as np


class PolyAdaptor:

    def fit(self, X, Y, degree):
        self._coef = np.polyfit(X, Y, degree)
        self.min_domain = X.min()
        self.max_domain = X.max()
        self.p = np.poly1d(self._coef)
        return self

    def predict(self, X):
        return self.p(X)

    def predict_inverse(self, Y):
        coef = self.p.coef.copy()
        coef[-1] -= Y
        roots = np.roots(coef)
        real_roots = roots[np.isreal(roots)].real

        real_roots_within_domain = real_roots[
            (real_roots >= self.min_domain) & (real_roots <= self.max_domain)
        ]
        if len(real_roots_within_domain) < 1:
            raise ValueError("no solution was found")
        if len(real_roots_within_domain) > 2:
            raise ValueError("to many roots where found")

        return real_roots_within_domain[-1]

    def dxdt(self):
        self.p = np.poly1d(np.polyder(self._coef))
        return self
