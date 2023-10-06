import matplotlib.pyplot as plt


def plottingLoss(list_loss):
    """"
    Plotting the loss of the model in function of the steps
    """
    plt.plot(range(len(list_loss)), list_loss)
    plt.xlabel('Number of Steps')
    plt.ylabel('MSE loss')
    plt.title('Loss Over Steps')
    plot_file_name = 'savedModels/lossPlot.png'
    plt.savefig(plot_file_name)
    # Display or save the plot
    plt.show()
    return