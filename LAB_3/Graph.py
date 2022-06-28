import matplotlib.pyplot as plt


def draw(y, title='Fitness Graph'):
    x = []
    # x axis values
    for i in range(len(y)):
        x.append(i)

    # corresponding y axis values
    # plotting the points
    plt.plot(x, y)

    # naming the x axis
    plt.xlabel('Iteration')
    # naming the y axis
    plt.ylabel('Min Fitness')

    # giving a title to my graph
    plt.title(title)

    # function to show the plot
    plt.show()
