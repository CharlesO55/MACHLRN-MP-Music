import torch.nn as nn
import torch.nn.init
import torch.nn.functional as F

class NeuralNetwork(nn.Module):

    def __init__(self,
                 input_size,
                 num_classes,
                 list_hidden,
                 activation='sigmoid'):
        """Class constructor for NeuralNetwork

        Arguments:
            input_size {int} -- Number of features in the dataset
            num_classes {int} -- Number of classes in the dataset
            list_hidden {list} -- List of integers representing the number of
            units per hidden layer in the network
            activation {str, optional} -- Type of activation function. Choices
            include 'sigmoid', 'tanh', and 'relu'.
        """
        super(NeuralNetwork, self).__init__()

        self.input_size = input_size
        self.num_classes = num_classes
        self.list_hidden = list_hidden
        self.activation = activation

    def create_network(self):
        """Creates the layers of the neural network.
        """
        layers = []

        # Append a torch.nn.Linear layer to the
        # layers list with correct values for parameters in_features and
        # out_features. This is the first layer of the network.
        layers.append(nn.Linear(in_features=self.input_size, out_features=self.list_hidden[0]))
        
        # Append the activation layer by calling
        # the self.get_activation() function.
        #layers.append(None)
        layers.append(self.get_activation())
        
        # Iterate over other hidden layers just before the last layer
        for i in range(len(self.list_hidden) - 1):

            # Append a torch.nn.Linear layer to
            # the layers list according to the values in self.list_hidden.
            layers.append(nn.Linear(in_features=self.list_hidden[i], out_features=self.list_hidden[i+1]))
            
            # Append the activation layer by
            # calling the self.get_activation() function.
            layers.append(self.get_activation(self.activation))

        # Append a torch.nn.Linear layer to the
        # layers list with correct values for parameters in_features and
        # out_features. This is the last layer of the network.
        layers.append(nn.Linear(in_features=self.list_hidden[-1], out_features=self.num_classes))
        
        layers.append(nn.Softmax(dim=1))
        self.layers = nn.Sequential(*layers)

    def init_weights(self):
        """Initializes the weights of the network. Weights of a
        torch.nn.Linear layer should be initialized from a normal
        distribution with mean 0 and standard deviation 0.1. Bias terms of a
        torch.nn.Linear layer should be initialized with a constant value of 0.
        """
        torch.manual_seed(2)

        # For each layer in the network
        for module in self.modules():

            # If it is a torch.nn.Linear layer
            if isinstance(module, nn.Linear):

                # Initialize the weights of the torch.nn.Linear layer
                # from a normal distribution with mean 0 and standard deviation
                # of 0.1.
                nn.init.normal_(module.weight, mean=0.0, std=0.1)

                # Initialize the bias terms of the torch.nn.Linear layer
                # with a constant value of 0.
                nn.init.constant_(module.bias, 0.0)


    def get_activation(self,
                       mode='sigmoid'):
        """Returns the torch.nn layer for the activation function.

        Arguments:
            mode {str, optional} -- Type of activation function. Choices
            include 'sigmoid', 'tanh', and 'relu'.

        Returns:
            torch.nn -- torch.nn layer representing the activation function.
        """
        activation = nn.Sigmoid()

        if mode == 'tanh':
            activation = nn.Tanh()

        elif mode == 'relu':
            activation = nn.ReLU(inplace=True)

        return activation

    def forward(self,
                x,
                verbose=False):
        """Forward propagation of the model, implemented using PyTorch.

        Arguments:
            x {torch.Tensor} -- A Tensor of shape (N, D) representing input
            features to the model.
            verbose {bool, optional} -- Indicates if the function prints the
            output or not.

        Returns:
            torch.Tensor, torch.Tensor -- A Tensor of shape (N, C) representing
            the output of the final linear layer in the network. A Tensor of
            shape (N, C) representing the probabilities of each class given by
            the softmax function.
        """

        # For each layer in the network
        for i in range(len(self.layers) - 1):

            # Call the forward() function of the layer
            # and return the result to x.
            x = self.layers[i](x)

            if verbose:
                # Print the output of the layer
                print('Output of layer ' + str(i))
                print(x, '\n')

        # Apply the softmax function
        probabilities = self.layers[-1](x)

        if verbose:
            print('Output of layer ' + str(len(self.layers) - 1))
            print(probabilities, '\n')

        return x, probabilities

    def predict(self,
                probabilities):
        """Returns the index of the class with the highest probability.

        Arguments:
            probabilities {torch.Tensor} -- A Tensor of shape (N, C)
            representing the probabilities of N instances for C classes.

        Returns:
            torch.Tensor -- A Tensor of shape (N, ) contaning the indices of
            the class with the highest probability for N instances.
        """

        # Return the index of the class with the highest probability
        _,output = torch.max(probabilities, dim=1)
        return output
