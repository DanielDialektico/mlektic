import matplotlib.pyplot as plt
from IPython.display import clear_output
from typing import List, Union, Tuple

def plot_dynamic_cost(cost_history: List[Union[float, int]], title: str = "Training Cost over Iterations", xlabel: str = "Iterations", ylabel: str = "Cost", 
                      title_size: int = 13, label_size: int = 10, style: str = 'bmh', point_color: str = 'blue', line_color: str = 'black', pause_time: float = 0.1, dim: Tuple = (10, 6)) -> None:
    """
    Generates a dynamic plot of the cost history during training,
    showing each point as it is added.

    Args:
        cost_history (List[Union[float, int]]): List of cost values recorded during training.
        title (str, optional): Title of the plot. Default is "Training Cost over Iterations".
        xlabel (str, optional): Label for the x-axis. Default is "Iterations".
        ylabel (str, optional): Label for the y-axis. Default is "Cost".
        title_size (int, optional): Font size of the title. Default is 13.
        label_size (int, optional): Font size of the axis labels. Default is 10.
        style (str, optional): Style of the plot. Default is 'bmh'.
        point_color (str, optional): Color of the points. Default is 'blue'.
        line_color (str, optional): Color of the line. Default is 'black'.
        pause_time (float, optional): Pause time between updates in seconds. Default is 0.1.
        dim (Tuple, optional): Dimensions of the plot (width, height). Default is (10, 6).

    Returns:
        None
    """

    plt.figure(figsize=dim)
    plt.style.use(style)
    
    total_iterations = len(cost_history)
    
    for i in range(1, total_iterations + 1):
        clear_output(wait=True)
        plt.plot(range(i), cost_history[:i], marker='o', color=point_color, linestyle='-', linewidth=2, markersize=5, label='Cost' if i == 1 else "")
        plt.plot(range(i), cost_history[:i], color=line_color, linewidth=2)
        plt.title(title, fontsize=title_size, fontweight='bold')
        plt.xlabel(xlabel, fontsize=label_size)
        plt.ylabel(ylabel, fontsize=label_size)
        plt.xlim(0, total_iterations - 1)
        plt.ylim(0, max(cost_history) * 1.1)
        plt.grid(True) 
        if i == 1:
            plt.legend()
        
        plt.draw()
        plt.pause(pause_time)
    
    plt.show()

def plot_cost(cost_history: List[Union[float, int]], epoch_points: bool = False, title: str = "Training Cost over Iterations", xlabel: str = "Iterations", ylabel: str = "Cost", 
                    title_size: int = 13, label_size: int = 10, style: str = 'bmh', point_color: str = 'blue', line_color: str = 'blue', dim: Tuple = (7, 5)) -> None:
    """
    Generates a static plot of the cost history during training.

    Args:
        cost_history (List[Union[float, int]]): List of cost values recorded during training.
        epoch_points (bool, optional): Whether to show points at each epoch. Default is False.
        title (str, optional): Title of the plot. Default is "Training Cost over Iterations".
        xlabel (str, optional): Label for the x-axis. Default is "Iterations".
        ylabel (str, optional): Label for the y-axis. Default is "Cost".
        title_size (int, optional): Font size of the title. Default is 13.
        label_size (int, optional): Font size of the axis labels. Default is 10.
        style (str, optional): Style of the plot. Default is 'bmh'.
        point_color (str, optional): Color of the points. Default is 'blue'.
        line_color (str, optional): Color of the line. Default is 'black'.
        dim (Tuple, optional): Dimensions of the plot (width, height). Default is (7, 5).

    Returns:
        None
    """

    plt.figure(figsize=dim)
    plt.style.use(style)

    plt.plot(range(len(cost_history)), cost_history, color=line_color, linewidth=2, label='Cost')
    
    if epoch_points:
        plt.plot(range(len(cost_history)), cost_history, marker='o', color=point_color, linestyle='', markersize=5)

    plt.title(title, fontsize=title_size, fontweight='bold')
    plt.xlabel(xlabel, fontsize=label_size)
    plt.ylabel(ylabel, fontsize=label_size)
    plt.xlim(0, len(cost_history) - 1)
    plt.ylim(0, max(cost_history) * 1.1)
    plt.grid(True)
    plt.legend()
    
    plt.show()
