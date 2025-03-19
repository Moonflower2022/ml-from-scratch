import numpy as np
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.datasets import mnist
import time

from base import ForwardCaller, LinearLayer, sigmoid, Dsigmoid, log_loss

def one_hot_encode(labels, num_classes=10):
    encoded = np.zeros((labels.size, num_classes))
    for i, label in enumerate(labels):
        encoded[i, label] = 1
    return encoded

class TestModel(ForwardCaller):
    def __init__(self, input_size, output_size):
        self.layers = [
            LinearLayer(input_size, 100, (-.1, .01), (-.1, .1)), 
            sigmoid, 
            LinearLayer(100, output_size, (-.1, .01), (-.1, .1)), 
            sigmoid
        ]

        # values should be X, X1, A1, X2, A2, Y
        self.gradient_functions = [
            {
                "weights": lambda *values: (self.layers[2].weights.T @ (values[-2] - values[-1]) * Dsigmoid(values[-5])) @ values[-6].T, 
                "biases": lambda *values: np.sum(self.layers[2].weights.T @ (values[-2] - values[-1]) * Dsigmoid(values[-5]), axis=1)
            }, 
            None, 
            {
                "weights": lambda *values: (values[-2] - values[-1]) @ values[-4].T, 
                "biases": lambda *values: np.sum(values[-2] - values[-1], axis=1)
            }, 
            None
        ]

        # structure:
        # input: X
        # X1 = M1 * X + B1
        # X2 = tanh(X1)
        # l = (X2 - Y)^2

        # f(x) = derivative of activation function
        # dl/dB1 = sum(2 * (X2 - Y) * f(X1), axis = 0) (row sum)
        # dl/dW1 = 2 * (X2 - Y) * f(X1) * X

    def forward(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x

    def step(self, inputs, targets, learning_rate):
        intermediate_outputs = [inputs]
        x = inputs
        for layer in self.layers:
            intermediate_outputs.append(layer(x))
            x = layer(x)
        Y = targets

        loss = log_loss(x, Y) / Y.shape[1]
        for i, gradient_function in enumerate(self.gradient_functions):
            if gradient_function:
                for key, function in gradient_function.items():
                    self.layers[i].__dict__[key] -= learning_rate * function(*intermediate_outputs, Y)

        # o is output length
        # i is input length
        # b is batch size
        # (o, b) * (o, b) @ (i, b).T
        # (o, b) * (o, b) @ (b, i)
        # (o, i)

        # outputs, loss, gradient
        return x, loss


def test_model():
    input_size = 3
    output_size = 2

    inputs = np.array([[1, 2, 3], [4, 5, 6]]).T
    targets = np.array([[0, 1], [1, 0]]).T

    print(inputs.shape, targets.shape)

    model = TestModel(input_size, output_size)

    for _ in range(10000):
        outputs, loss = model.step(inputs, targets, 0.01)
        print("loss:", loss)
        print("outputs:", outputs)

def train_mnist():
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

    X_train = X_train.T
    y_train = y_train.T

    X_test = X_test.T
    y_test = y_test.T

    model = TestModel(X_train.shape[0], y_train.shape[0])

    for i in range(100):
        outputs, loss = model.step(X_train, y_train, 0.01)
        print(f"[epoch {i}] loss:", loss)
    
    for i in range(100):
        outputs, loss = model.step(X_train, y_train, 0.001)
        print(f"[epoch {i}] loss:", loss)

    outputs, _ = model.step(X_test, y_test, 0)
    outputs = outputs.T
    y_test = y_test.T

    num_correct = 0
    for output, target in zip(outputs, y_test):
        if target[np.argmax(output)] == 1:
            num_correct += 1

    print("accuracy:", num_correct / y_test.shape[0])

def new_train_mnist():
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    train_cutoff = 2000
    test_cutoff = 200

    train_images = train_images.astype(np.float32)[:train_cutoff] / 255.0
    test_images = test_images.astype(np.float32)[:test_cutoff] / 255.0

    train_labels = one_hot_encode(train_labels[:train_cutoff])
    test_labels = one_hot_encode(test_labels[:test_cutoff])

    train_images = train_images.reshape(-1, 28*28)
    test_images = test_images.reshape(-1, 28*28)


    train_images = train_images.T
    test_images = test_images.T

    train_labels = train_labels.T
    test_labels = test_labels.T

    print("Train shape:", train_images.shape, "Test shape:", test_images.shape)

    model = TestModel(train_images.shape[0], train_labels.shape[0])

    start_time = time.time()

    for i in range(150):
        outputs, loss = model.step(train_images, train_labels, 0.002)
        print(f"[epoch {i}] loss:", loss)
    
    for i in range(150):
        outputs, loss = model.step(train_images, train_labels, 0.001)
        print(f"[epoch {i}] loss:", loss)

    print("training time (s):", time.time() - start_time)

    outputs = model(test_images)
    outputs = outputs.T
    test_labels = test_labels.T

    num_correct = 0
    for output, target in zip(outputs, test_labels):
        if target[np.argmax(output)] == 1:
            num_correct += 1

    print("accuracy:", num_correct / test_labels.shape[0])

if __name__ == "__main__":
    new_train_mnist()