import numpy as np

from base import ForwardCaller, LinearLayer, leaky_relu, Dleaky_relu
import torch

class TestModel(ForwardCaller):
    def __init__(self, input_size, output_size):
        self.layers = [LinearLayer(input_size, output_size), leaky_relu]
        self.gradient_functions = [{"weights": lambda *values: 2 * (values[-2] - values[-1]) * Dleaky_relu(values[-3]) @ values[-4].T, "biases": lambda *values: np.sum(2 * (values[-2] - values[-1]) * Dleaky_relu(values[-3]), axis=1)}, None]

        # structure: 
        # X1 = M1 * X + B1
        # X2 = tanh(X1)
        # l = (X2 - Y)^2

        # f(x) = derivative of activation function
        # dl/dB1 = sum(2 * (X2 - Y) * f(X1), axis = 0) (row sum)
        # dl/dW1 = 2 * (X2 - Y) * f(X1) * X

    def step(self, inputs, targets, learning_rate):
        intermediate_outputs = [inputs]
        x = inputs
        for layer in self.layers:
            intermediate_outputs.append(layer(x))
            x = layer(x)
        Y = targets

        loss = np.sum(np.power(x - Y, 2)) / Y.shape[0] / Y.shape[1]
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

def test_with_pytorch():
    inputs = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    targets = np.array([[0, 1], [1, 0]], dtype=np.float32)

    model = torch.nn.Linear(3, 2)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_function = torch.nn.MSELoss()

    num_epochs = 10000

    for _ in range(num_epochs):
        optimizer.zero_grad()

        outputs = model(torch.tensor(inputs))
        loss = loss_function(outputs, torch.tensor(targets))
        loss.backward()
        optimizer.step()

    print(loss)

if __name__ == "__main__":
    input_size = 3
    output_size = 2

    inputs = np.array([[1, 2, 3], [4, 5, 6]]).T
    targets = np.array([[0, 1], [1, 0]]).T

    model = TestModel(input_size, output_size)

    for _ in range(10000):
        outputs, loss = model.step(inputs, targets, 0.001)
        print("loss:", loss)
