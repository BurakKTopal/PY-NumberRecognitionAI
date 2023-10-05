import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def plotting(list_nums, list_probs):
    """"
    Plotting the certainty of number guesses
    """
    product_strings = list_nums[0]
    product_prob = np.array(list_probs[0])

    for index in range(1, len(list_nums)):
        # Mimicking tensorproduct between two lists containing strings as input
        list_1 = list_nums[index]
        product_strings = [num1 + num2 for num1 in product_strings for num2 in list_1]

    for index in range(0, len(product_strings)):
        product_strings[index] = int(product_strings[index])  # Reconverting the number strings back to integers


    for index in range(1, len(list_probs)):
        matrix = list_probs[index]
        tensor = np.array(matrix)
        product_prob = np.kron(product_prob, tensor)  #Taking the tensor product to calculate probs for each possibility

    for index in range(len(product_prob)):
        product_prob[index] = round(product_prob[index]*100, 2) # Turning the probabilities in percentages

    plt.scatter(product_strings, product_prob)  # Scatter plot

    for x, y in zip(product_strings, product_prob):
        if y > 5:
            plt.text(x, y, f'({x}, {y:.2f})', ha='left')  # Labeling points that have a certainty bigger then 5%

    plt.xlabel('Number')
    plt.ylabel('Certainty(%)')
    plt.title('Certainty Plot')
    plot_file_name = 'static/plots/certainty_plot.png'
    plt.savefig(plot_file_name)
    plt.close()
    return
