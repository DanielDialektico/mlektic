"""Optimization utilities for Google Colab and heavy environments."""

from IPython.display import HTML

def show_optimized(fig):
    """
    Convierte una figura interactiva de Plotly a un iFrame HTML estático.
    Ideal para Google Colab, saltando el pesado renderizador ipywidgets 
    para animaciones 3D con decenas de miles de vértices.
    """
    # Usar el height del layout o un default
    fig_height = fig.layout.height if fig.layout.height else 600
    wrapper_height = fig_height + 40  # +40 para darle respiro y no cortar el slider

    html_str = fig.to_html(
        include_plotlyjs="cdn", 
        full_html=False, 
        auto_play=False, 
        include_mathjax="cdn"
    )
    return HTML(f'<div style="height: {wrapper_height}px; width: 100%; overflow: hidden;">{html_str}</div>')

__all__ = ["show_optimized"]
