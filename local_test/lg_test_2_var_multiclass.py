import numpy as np
import plotly.io as pio
from sklearn.linear_model import SGDClassifier
from sklearn.datasets import make_classification
from mlektic import visualize_logistic

X, y = make_classification(
    n_samples=200, 
    n_features=2, 
    n_informative=2, 
    n_redundant=0, 
    n_repeated=0, 
    n_classes=3, 
    n_clusters_per_class=1, 
    random_state=42
)

model = SGDClassifier(
    loss="log_loss",
    learning_rate="constant",
    eta0=0.01,
    max_iter=200,
    random_state=42
)

model.fit(X, y)

fig = visualize_logistic(
    model, X, y,
    steps=60,
    show_loss=True,
    frame_duration=80,
    title="Test Multiclass 2D",
    dec=3
)

fig.write_html("test_output_multiclass_2d.html", auto_play=False)
print("Generado test_output_multiclass_2d.html exitosamente")
