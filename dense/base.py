import numpy as np

class ForwardCaller:
    def __call__(self, inputs):
        return self.forward(inputs)


class WeightLayer(ForwardCaller):
    num_parameter_groups = 1

    def __init__(self,
        input_size,
        output_size,
        initialization_range=(-1, 1)):

        self.input_size = input_size  # int
        self.output_size = output_size  # int
        self.weights_shape = (output_size, input_size)
        self.weights = np.random.uniform(
            *initialization_range, size=(output_size, input_size)
        )

    def forward(self, x):
        return self.weights @ x
    
    def get_parameters(self):
        return (self.weights,)

    def get_shapes(self):
        return (self.weights_shape,)
        


class LinearLayer(ForwardCaller):
    num_parameter_groups = 2

    def __init__(
        self,
        input_size,
        output_size,
        weight_initialization_range=(-1, 1),
        bias_initialization_range=(-1, 1),
    ):
        self.input_size = input_size  # int
        self.output_size = output_size  # int
        self.weights_shape = (output_size, input_size)
        self.biases_shape = (output_size,)
        self.weights = np.random.uniform(
            *weight_initialization_range, size=(output_size, input_size)
        )
        self.biases = np.random.uniform(*bias_initialization_range, size=(output_size,))

        self.parameter_groups = [self.weights, self.biases]

    def forward(self, inputs):
        return self.weights @ inputs + self.biases.reshape(-1, 1)

    def get_parameters(self):
        return (self.weights, self.biases)

    def get_shapes(self):
        return (self.weights_shape, self.biases_shape)
    

def mean_squared_loss(outputs, target):
    return np.sum(np.power(outputs - target, 2))

def log_loss(outputs, target):
    # cant have negative inputs
    return np.sum(-(target * np.log(outputs) + (1 - target) * np.log(1 - outputs)))

def softmax(x):
    return np.exp(x) / sum(np.exp(x))

def Dtanh(x):
    return (1 - np.power(np.tanh(x), 2))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def Dsigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

def relu(x):
    return np.where(x > 0, x, 0)

def Drelu(x):
    return np.where(x > 0, 1, 0)

def leaky_relu(x, alpha=0.1):
    return np.where(x > 0, x, x * alpha)

def Dleaky_relu(x, alpha=0.1):
    return np.where(x > 0, 1, alpha)