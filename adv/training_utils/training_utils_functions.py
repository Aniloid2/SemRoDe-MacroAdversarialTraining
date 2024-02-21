class ExponentialMovingAverage:
    def __init__(self, weight=0.3):
        self._weight = weight
        self.reset()

    def update(self, x):
        self._x += x
        self._i += 1
        self._cov = ((self._i - 1) * self._cov + (x - self._ave) *
                     (x - self._x / (self._i + 1e-13))) / (self._i + 1e-13)
        self._ave = self._x / (self._i + 1e-13)



    def reset(self):
        self._x = 0
        self._i = 0
        self._cov = 0
        self._ave = 0

    def get_metric(self):
        return [self._ave, self._cov]
