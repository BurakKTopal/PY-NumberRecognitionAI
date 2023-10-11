import matplotlib.pyplot as plt


def plottingLoss(list_mean_loss):
    """"
    Plotting the loss of the model in function of the steps
    """
    plt.plot(range(len(list_mean_loss)), list_mean_loss)
    plt.xlabel('Number of steps(per 100)')
    expression = r'$10^{-3}$'
    plt.ylabel('Average MSE loss(' + expression + ')')
    plt.title('Average loss Over Steps')
    plot_file_name = 'savedModels/lossPlot' + str(int(len(list_mean_loss)//100)) + '.png'
    plt.savefig(plot_file_name)
    # Display or save the plot
    plt.show()
    return
