import numpy as np
import pickle


class neuralNetwork4Layers():
    """"
    Neural network for the image processing
    """

    def __init__(self):
        self.input_size = 784
        self.hidden_one_size = self.input_size // 3 * 2
        self.hidden_two_size = self.hidden_one_size//3 * 2
        self.output_size = 10   # Ten numbers and 4 operators
        self.weights_i_h1 = np.random.uniform(-0.5, 0.5, (self.hidden_one_size, self.input_size))
        self.weights_h1_h2 = np.random.uniform(-0.5, 0.5, (self.hidden_two_size, self.hidden_one_size))
        self.weights_h2_o = np.random.uniform(-0.5, 0.5, (self.output_size, self.hidden_two_size))
        self.target = None
        self.input = None
        self.hidden1 = None
        self.hidden2 = None
        self.output = None
        self.learning_rate = 0.1
        self.bias_i_h1 = np.zeros((self.hidden_one_size, 1))
        self.bias_h1_h2 = np.zeros((self.hidden_two_size, 1))
        self.bias_h2_o = np.zeros((self.output_size, 1))

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
        return x * (1 - x)

    def MSELoss(self, output, target):
        """"
        L2-Norm, not used in this model, yet important to calculate
        """
        return sum((output - target) ** 2) * 1000

    def forward(self, input, target):
        """"
        forward
        """
        self.input = np.array(input)  # making np array of input, size: 784 x 1

        self.hidden1 = self.sigmoid(
            self.weights_i_h1 @ self.input + self.bias_i_h1)  # Passing values through activation function

        self.hidden2 = self.sigmoid(
            self.weights_h1_h2 @ self.hidden1 + self.bias_h1_h2)

        self.output = self.sigmoid(
            self.weights_h2_o @ self.hidden2 + self.bias_h2_o)  # Output of neural network

        self.delta = self.output - target

        self.loss = self.MSELoss(self.output, target)

        return

    def backward(self):
        """"
        Updating the weight matrices and the bias vectors according to the MSE loss function
        """
        """"Updating the weight matrix connecting the second hidden layer to the output layer"""
        delta_o = (self.delta*self.output*(1-self.output))
        div_L_o_h2 = delta_o @ np.transpose(self.hidden2)   # Calculating grad from output to hidden
        self.weights_h2_o += -self.learning_rate*div_L_o_h2

        bias_hidden2_div = delta_o
        self.bias_h2_o += -self.learning_rate*bias_hidden2_div

        #https://365datascience.com/trending/backpropagation/
        """Updating weight matrix connecting second hidden layer to the first one"""
        A_matrix = self.delta*self.output*(1-self.output)
        B_matrix = np.transpose(A_matrix)@self.weights_h2_o
        C_matrix = np.transpose(B_matrix)*(self.hidden2*(1-self.hidden2))
        div_L_h1_h2 = C_matrix@np.transpose(self.hidden1)
        self.weights_h1_h2 += -self.learning_rate*div_L_h1_h2

        bias_h1_h2_div = C_matrix
        self.bias_h1_h2 += -self.learning_rate*bias_h1_h2_div


        """"Updating the weight matrix connecting input layer to the first hidden layer"""
        A_matrix = self.delta*self.output*(1-self.output)
        B_matrix = np.transpose(A_matrix)@self.weights_h2_o
        C_matrix = np.transpose(B_matrix)*self.hidden2*(1-self.hidden2)
        D_matrix = np.transpose(C_matrix)@self.weights_h1_h2
        E_matrix = np.transpose(D_matrix)*self.hidden1*(1-self.hidden1)
        div_L_i_h1 = E_matrix@np.transpose(self.input)

        self.weights_i_h1 += -self.learning_rate*div_L_i_h1

        bias_i_h1 = E_matrix
        self.bias_i_h1 += -self.learning_rate*bias_i_h1

        return

    def save(self):
        """"
        Saving the model
        """
        # Define the file path where you want to save the matrices
        file_path = "model4Layers.pkl"

        # Save the matrices to a file
        with open(file_path, 'wb') as file:
            pickle.dump((self.weights_i_h1, self.weights_h1_h2, self.weights_h2_o), file)
        return

    def load(self, file_path):
        """"
        Loading the model, using pkl's
        """
        # Load the matrices from the file
        with open(file_path, 'rb') as file:
            self.weights_i_h1, self.weights_h1_h2, self.weights_h2_o = pickle.load(file)
        return self.weights_i_h1, self.weights_h1_h2, self.weights_h2_o
