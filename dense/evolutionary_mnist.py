import numpy as np
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from evolutionary import EvolutionaryModel, log_loss, LinearLayer
import time

mnist = load_digits()
X = mnist.data
y = mnist.target

scaler = StandardScaler()
X = scaler.fit_transform(X)

y_one_hot = np.zeros((y.size, 10))
y_one_hot[np.arange(y.size), y] = 1

cutoff = int(len(X) / 2)

X_train, X_test = X[:cutoff], X[cutoff:]
y_train, y_test = y_one_hot[:cutoff], y_one_hot[cutoff:]

if __name__ == "__main__":
    print(X_train.shape)
    print(y_train.shape)

    layers = [
        LinearLayer(X_train.shape[1], 128),
        np.tanh,
        LinearLayer(128, y_train.shape[1]),
        np.tanh,
        lambda x: np.exp(x) / np.sum(np.exp(x))
    ]

    model = EvolutionaryModel(layers, num_children=10, mutation_probability=0.2, step_size=0.1)

    start_time = time.time()

    model.train(X_train, y_train, log_loss, epochs=1000)
    model.step_size = 0.01
    model.train(X_train, y_train, log_loss, epochs=1000)

    print("Training Time (s):", time.time() - start_time)

    num_correct = 0
    for i, x in enumerate(X_test):
        if np.argmax(model.forward(x)) == np.argmax(y_test[i]):
            num_correct += 1
    print("accuracy:", num_correct / len(X_test))