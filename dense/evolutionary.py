import numpy as np
import random

from base import ForwardCaller, LinearLayer, mean_squared_loss

class EvolutionaryModel(ForwardCaller):
    def __init__(self, layers, num_children=10, mutation_probability=0.8, step_size=0.001):
        self.layers = layers

        self.num_children = num_children
        self.mutation_probability = mutation_probability
        self.step_size = step_size

    def forward(self, inputs, layers=None):
        layers = self.layers if not layers else layers
        x = inputs
        for layer in layers:
            x = layer(x)
        return x

    def get_all_parameters(self):
        all_parameters = tuple()
        for layer in self.layers:
            if isinstance(layer, LinearLayer):
                all_parameters += layer.get_parameters()

        return all_parameters

    def get_all_shapes(self):
        all_shapes = tuple()
        for layer in self.layers:
            if isinstance(layer, LinearLayer):
                all_shapes += layer.get_shapes()

        return all_shapes

    def optimize(self, inputs, target, loss_function):
        all_shapes = self.get_all_shapes()
        mutations = [self.generate_mutation(all_shapes) for _ in range(self.num_children)]

        best_mutation_loss = float('inf')
        best_mutation = None
        
        for mutation in mutations:
            self.add_mutation(mutation)
            total_loss = 0
            # calculate loss for each input-target pair
            for input_data, target_data in zip(inputs, target):
                outputs = self.forward(input_data)
                total_loss += loss_function(outputs, target_data)
            
            if total_loss < best_mutation_loss:
                best_mutation_loss = total_loss
                best_mutation = mutation
            self.add_mutation(mutation, inverse=True)

        self.add_mutation(best_mutation)

    def generate_mutation(self, all_shapes):
        return [
            np.random.uniform(-self.step_size, self.step_size, parameter_shape) if random.random() < self.mutation_probability else 0
            for parameter_shape in all_shapes
        ]

    def add_mutation(self, mutation, inverse=False):
        i = 0
        for layer in self.layers:
            if isinstance(layer, LinearLayer):
                for j in range(layer.num_parameter_groups):
                    layer.parameter_groups[j] += (
                        mutation[i] if inverse else -mutation[i]
                    )
                    i += 1

    def train(self, inputs, expected_outputs, loss_function, epochs=1000):
        for epoch in range(epochs):
            self.optimize(inputs, expected_outputs, loss_function)
            if epoch % 100 == 0:
                total_loss = 0
                for input_data, target_data in zip(inputs, expected_outputs):
                    outputs = self.forward(input_data)
                    total_loss += loss_function(outputs, target_data)
                print(f"Epoch {epoch}, Average Loss: {total_loss/len(inputs):.4f}")

if __name__ == "__main__":
    linear1 = LinearLayer(3, 4)
    activation1 = np.tanh
    linear2 = LinearLayer(4, 5)
    activation2 = np.tanh
    layers = [linear1, activation1, linear2, activation2]

    model = EvolutionaryModel(layers, num_children=100, step_size=0.01)
    inputs = np.array([[1, 2, 3], [4, 5, 6]])
    target = np.array([[0.5, 1, 0, 0, -1], [0.5, 1, 0, 0, 1]])
    model.train(inputs, target, mean_squared_loss)
    
    print("Final predictions:")
    for input_data in inputs:
        prediction = model(input_data)
        print(f"Input: {input_data}, Output: {prediction}")