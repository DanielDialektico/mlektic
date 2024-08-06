import pytest
import matplotlib.pyplot as plt
from mlektic.plot_utils.plot_utils import plot_dynamic_cost

def test_plot_dynamic_cost():
    # Test data
    cost_history = [10, 8, 6, 4, 2, 1]
    
    # Set matplotlib to use the Agg backend
    plt.switch_backend('Agg')

    # Call the plot function
    plot_dynamic_cost(
        cost_history,
        title="Test Cost Plot",
        xlabel="Test Iterations",
        ylabel="Test Cost",
        title_size=15,
        label_size=12,
        style='fast',
        point_color='red',
        line_color='green',
        pause_time=0.01
    )
    
    # Save the plot to a file
    plt.savefig('./test_plot_dynamic_cost.png')
    
    # Check if the file exists and has a reasonable size
    import os
    assert os.path.exists('./test_plot_dynamic_cost.png')
    assert os.path.getsize('./test_plot_dynamic_cost.png') > 0
    
if __name__ == "__main__":
    pytest.main()
