import numpy as np
from scipy.special import expit

class VanillaNeuralNet():
    def __init__(self, W=None, V=None):
        if W:
            self.W =  W
        else:
            self.W = np.random.randn(26, 201) / 201.0
        if V:
            self.V = V
        else:
            self.V = np.random.randn(200, 785) / 785.0
        self.steps = 0


    def train(self, X, Y, epochs=5, W_learn=(0.5, 0.7, 1), V_learn=(0.9, 0.8, 1), name="weights"):
        count, epoch = 0, 0
        size = X.shape[0]
        idx = np.random.permutation(size)
        eW, eV = W_learn[0], V_learn[0]
        for i in range(int(epochs * size)):
            try:
                dW = np.zeros((26, 201))
                dV = np.zeros((200, 785))
                x, y = X[idx[count]], Y[idx[count]]
                h = np.append(np.tanh(self.V.dot(x)), 1)
                z = expit(self.W.dot(h))
                dz = (z - y)
                dW += np.outer(dz, h)
                dV += np.outer(np.sum(dz.reshape(26, 1) * self.W[...,:-1], axis=0) * (1-h[:-1]**2), x)
                if (count == size - 1):
                    idx = np.random.permutation(size)
                    count = -1
                    epoch += 1
                    if (epoch % W_learn[2] == 0):
                        eW *= W_learn[1]
                    if (epoch % V_learn[2] == 0):
                        eV *= V_learn[1]
                    print("Finished epoch " + str(epoch))
                count += 1
                self.W -= eW * dW
                self.V -= eV * dV
                self.steps += 1
            except KeyboardInterrupt:
                break
        np.savez(name + str(self.steps), V=self.V, W=self.W)

    def predict(self, X):
        pred = []
        for i in range(X.shape[0]):
            pred.append(np.argmax(expit(self.W.dot(np.append(np.tanh(self.V.dot(X[i])), 1)))))
        return pred
