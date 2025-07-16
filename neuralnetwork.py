import numpy as np


class NeuralNetwork:
    def __init__(
            self,
            input_size,
            hidden_size,
            output_size,
    ) -> None:
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        xavier_init_sigma = np.sqrt(
            2 / (input_size + output_size)
        )

        self.weights_input2hidden = np.random.normal(
            scale=xavier_init_sigma,
            size=(hidden_size, input_size)
        )
        self.bias_input2hidden = np.random.normal(
            scale=xavier_init_sigma,
            size=(hidden_size,)
        )

        self.weights_hidden2output = np.random.normal(
            scale=xavier_init_sigma,
            size=(output_size, hidden_size)
        )
        self.bias_hidden2output = np.random.normal(
            scale=xavier_init_sigma,
            size=(output_size,)
        )

    def _act(self, x):
        return x * (x > 0)

    def _act_derivative(self, x):
        return 1 * (x > 0)
    
    def _softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum(axis=0)
    
    def _softmax_derivative(self, x):
        exp_x = np.exp(x - np.max(x))
        sum_exp_x = exp_x.sum(axis=0)
        return (exp_x * sum_exp_x + exp_x * exp_x) / (sum_exp_x ** 2)

    def forward(self, x):
        self.z1 = self.weights_input2hidden @ x + self.bias_input2hidden
        self.a1 = self._act(self.z1)

        self.z2 = self.weights_hidden2output @ self.a1 + self.bias_hidden2output
        self.a2 = self._softmax(self.z2)

        return self.a2
    
    def _dropout_forward(self, x, r1, r2, dropout_p):
        x = x * r1
        self.z1 = self.weights_input2hidden @ x / dropout_p + self.bias_input2hidden
        self.a1 = self._act(self.z1)

        self.a1 = self.a1 * r2
        self.z2 = self.weights_hidden2output @ self.a1 / dropout_p + self.bias_hidden2output

        self.a2 = self._softmax(self.z2)

        return self.a2

    def _accuracy(self, data, target):
        n_correct = 0
        n_samples = len(data)

        for sample, expected in zip(data, target):
            res = self.forward(sample)
            prediction = np.argmax(res)
            if expected[prediction] == 1:
                n_correct += 1

        return n_correct / n_samples

    def _loss(self, data, target):
        loss = 0
        for x, y in zip(data, target):
            y_pred = np.clip(self.forward(x), 1e-12, 1 - 1e-12)
            loss -= np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        return loss / len(data)

    def fit(self, data, target, epochs=100, lr=0.1, momentum_beta=0.9, dorpout_p = 0.9):
        n_samples = len(data)

        momentum_v_weights_input2hidden = np.zeros_like(self.weights_input2hidden)
        momentum_v_bias_input2hidden = np.zeros_like(self.bias_input2hidden)
        momentum_v_weights_hidden2output = np.zeros_like(self.weights_hidden2output)
        momentum_v_bias_hidden2output = np.zeros_like(self.bias_hidden2output)

        for epoch in range(epochs):
            r1 = np.random.binomial(1, 1 - dorpout_p, (self.input_size))
            r2 = np.random.binomial(1, 1 - dorpout_p, (self.hidden_size))

            shift_weights_input2hidden = np.zeros_like(self.weights_input2hidden)
            shift_bias_input2hidden = np.zeros_like(self.bias_input2hidden)
            shift_weights_hidden2output = np.zeros_like(self.weights_hidden2output)
            shift_bias_hidden2output = np.zeros_like(self.bias_hidden2output)

            for sample, expected in zip(data, target):
                res = self._dropout_forward(sample, r1, r2, dorpout_p)

                delta_output = 2 * (res - expected) * self._softmax_derivative(self.z2)
                delta_hidden = delta_output @ self.weights_hidden2output * self._act_derivative(self.z1)

                shift_weights_hidden2output += np.outer(delta_output, self.a1)
                shift_bias_hidden2output += delta_output
                shift_weights_input2hidden += np.outer(delta_hidden, sample)
                shift_bias_input2hidden += delta_hidden
            
            shift_bias_input2hidden /= n_samples
            shift_bias_hidden2output /= n_samples
            shift_weights_hidden2output /= n_samples
            shift_weights_input2hidden /= n_samples

            def calculate_shift(v_t, grad):
                return momentum_beta * v_t + (1 - momentum_beta) * grad

            momentum_v_weights_input2hidden = calculate_shift(momentum_v_weights_input2hidden, shift_weights_input2hidden)
            momentum_v_bias_input2hidden = calculate_shift(momentum_v_bias_input2hidden, shift_bias_input2hidden)
            momentum_v_weights_hidden2output = calculate_shift(momentum_v_weights_hidden2output, shift_weights_hidden2output)
            momentum_v_bias_hidden2output = calculate_shift(momentum_v_bias_hidden2output, shift_bias_hidden2output)

            self.weights_input2hidden -= lr * momentum_v_weights_input2hidden
            self.bias_input2hidden -= lr * momentum_v_bias_input2hidden
            self.weights_hidden2output -= lr * momentum_v_weights_hidden2output
            self.bias_hidden2output -= lr * momentum_v_bias_hidden2output

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}, Cost: {self._loss(data, target):.4f}, Accuracy: {self._accuracy(data, target)}")

