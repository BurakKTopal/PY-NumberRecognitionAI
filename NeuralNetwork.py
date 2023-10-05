import numpy as np
import pickle


class neuralNetwork():
    """"
    Neural network for the image processing
    """
    def __init__(self):
        self.input_size = 784
        self.hidden_size = self.input_size//3 * 2
        self.output_size = 10
        self.weights_i_h = np.random.uniform(-0.5, 0.5, (self.hidden_size, self.input_size))
        self.weights_h_o = np.random.uniform(-0.5, 0.5, (self.output_size, self.hidden_size))
        self.target = None
        self.input = None
        self.hidden = None
        self.output = None
        self.learning_rate = 0.1
        self.bias_i_h = np.zeros((self.hidden_size,1))
        self.bias_h_o = np.zeros((self.output_size,1))


    def sigmoid(self, x):
        """"
        Defining the sigmoid function
        """
        return 1 / (1 + np.exp(-x))

    def sigmoidDerivative(self, x):
        """"
        :param x: the sigmoid function value!
        :return: the derivative of the sigmoid function
        """
        return x*(1-x)
    def L2Norm(self, output, target):
        """"
        L2-Norm, not used in this model, yet important to calculate
        """
        return sum((output - target)**2)

    def forward(self, input, target):
        """"
        forward
        """
        self.input = np.array(input) # making np array of input, size: 420 x 1
        self.hidden = self.sigmoid(self.weights_i_h @ self.input + self.bias_i_h)   # Passing values through activation function
        self.output = self.sigmoid(self.weights_h_o @ self.hidden + self.bias_h_o)  # Output of neural network
        self.delta = self.output - target
        self.loss = self.L2Norm(self.output, target)
        return

    def backward(self):
        """"
        Updating the weight matrices and the bias vectors according to the MSE loss function
        """
        """"Updating the weight matrix connecting the hidden layer to the output layer"""
        delta_o = (self.delta*self.output*(1-self.output))
        div_L_o_h = delta_o @ np.transpose(self.hidden)   # Calculating grad from output to hidden
        self.weights_h_o += -self.learning_rate*div_L_o_h

        bias_hidden_div = delta_o
        self.bias_h_o += -self.learning_rate*bias_hidden_div

        """"Updating the weight matrix connecting input layer to hidden layer"""
        Q_matrix = self.delta*self.output
        delta_h = np.transpose(np.transpose(Q_matrix)@self.weights_h_o)*(self.hidden*(1-self.hidden))
        div_L_h_i = delta_h@np.transpose(self.input)
        self.weights_i_h += -self.learning_rate*div_L_h_i

        bias_input_div = delta_h
        self.bias_i_h += -self.learning_rate*bias_input_div

        return

    def save(self):
        """"
        Saving the model
        """
        # Define the file path where you want to save the matrices
        file_path = "modelDEF.pkl"

        # Save the matrices to a file
        with open(file_path, 'wb') as file:
            pickle.dump((self.weights_i_h, self.weights_h_o), file)
        return

    def load(self, file_path):
        """"
        Loading the model, using pkl's
        """
        # Load the matrices from the file
        with open(file_path, 'rb') as file:
            self.weights_i_h, self.weights_h_o = pickle.load(file)
        return self.weights_i_h, self.weights_h_o
