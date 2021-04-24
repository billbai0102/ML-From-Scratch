class MSE(object):
    def __init__(self, y, y_hat):
        self.y = y
        self.y_hat = y_hat

    def loss(self):
        