import matplotlib.pyplot as plt

def plot_learning_curve(x, scores, figure_file):
    plt.plot(x, scores)
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('Learning Curve')
    plt.savefig(figure_file)
    plt.close()
