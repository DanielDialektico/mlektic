import matplotlib.pyplot as plt
from IPython.display import clear_output
import time
from typing import List, Union

def plot_dynamic_cost(cost_history: List[Union[float, int]], title: str = "Training Cost over Iterations", xlabel: str = "Iterations", ylabel: str = "Cost", 
                      title_size: int = 13, label_size: int = 10, style: str = 'bmh', point_color: str = 'blue', line_color: str = 'black', pause_time: float = 0.1) -> None:
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

    Returns:
        None
    """
    plt.figure(figsize=(10, 6))
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