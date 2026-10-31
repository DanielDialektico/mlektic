import numpy as np
import plotly.graph_objs as go
import pytest
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mlektic import explain_logistic_prediction, visualize_logistic
from mlektic.services.linear_history import fit_history_logistic
from mlektic.utils.probability import multiclass_probabilities
from mlektic.visualization.logistic.router import build_logistic_figure


def _make_dummy_classification_data(n_samples=50, n_features=1, n_classes=2):
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    if n_classes == 2:
        y = (X[:, 0] > 0).astype(int)
    else:
        y = np.random.randint(0, n_classes, size=n_samples)
    return X, y


@pytest.fixture
def trained_binary_1d():
    X, y = _make_dummy_classification_data(50, 1, 2)
    model = SGDClassifier(loss='log_loss', max_iter=10, random_state=42)
    model.fit(X, y)
    return model, X, y


@pytest.fixture
def trained_binary_2d():
    X, y = _make_dummy_classification_data(50, 2, 2)
    model = SGDClassifier(loss='log_loss', max_iter=10, random_state=42)
    model.fit(X, y)
    return model, X, y


@pytest.fixture
def trained_binary_nd():
    X, y = _make_dummy_classification_data(50, 4, 2)
    model = SGDClassifier(loss='log_loss', max_iter=10, random_state=42)
    model.fit(X, y)
    return model, X, y


@pytest.fixture
def trained_multiclass_1d():
    X, y = _make_dummy_classification_data(50, 1, 3)
    model = SGDClassifier(loss='log_loss', max_iter=10, random_state=42)
    model.fit(X, y)
    return model, X, y


@pytest.fixture
def trained_multiclass_2d():
    X, y = _make_dummy_classification_data(50, 2, 3)
    model = SGDClassifier(loss='log_loss', max_iter=10, random_state=42)
    model.fit(X, y)
    return model, X, y


@pytest.fixture
def trained_multiclass_nd():
    X, y = _make_dummy_classification_data(50, 5, 3)
    model = SGDClassifier(loss='log_loss', max_iter=10, random_state=42)
    model.fit(X, y)
    return model, X, y


@pytest.fixture
def trained_pipeline_binary_1d():
    X, y = _make_dummy_classification_data(50, 1, 2)
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('sgd', SGDClassifier(loss='log_loss', max_iter=10, random_state=42))
    ])
    model.fit(X, y)
    return model, X, y


@pytest.mark.parametrize("n_features", [1, 2, 5])
@pytest.mark.parametrize("n_classes", [2, 3])
def test_probability_views_do_not_show_sample_prediction_results(n_features, n_classes):
    X, y = _make_dummy_classification_data(70, n_features, n_classes)
    model = LogisticRegression(max_iter=1000, random_state=42).fit(X, y)
    figure = visualize_logistic(model, X, y, steps=3, detail="complete")

    for annotations in [figure.layout.annotations, *(frame.layout.annotations for frame in figure.frames)]:
        visible_math = " ".join(str(annotation.text) for annotation in annotations)
        assert r"\hat{y}" not in visible_math
        assert r"\hat p_{0}=" not in visible_math
        assert r"\hat p_{1}=" not in visible_math


@pytest.mark.parametrize("dimensions", [1, 2])
def test_complete_multiclass_glossary_explains_displayed_symbols(dimensions):
    rng = np.random.default_rng(70 + dimensions)
    X = rng.normal(size=(60, dimensions))
    y = np.where(X[:, 0] < -0.4, 0, np.where(X[:, 0] > 0.4, 2, 1))
    model = LogisticRegression(max_iter=1000).fit(X, y)

    figure = visualize_logistic(model, X, y, steps=3, detail="complete")
    glossary = " ".join(
        annotation.text
        for annotation in figure.layout.annotations
        if annotation.yref == "paper" and float(annotation.y) < 0
    )
    for symbol in (
        "Model notation",
        r"\Theta=[\mathbf w_0",
        r"\boldsymbol\theta_0=",
        r"z_k(\mathbf x)=",
        r"\hat p_k(\mathbf x)=",
        r"k=0,\ldots,K-1",
    ):
        assert symbol in glossary
        assert all(
            any(symbol in str(annotation.text) for annotation in frame.layout.annotations)
            for frame in figure.frames
        )
    assert figure.layout.margin.b >= 300


class TestVisualizeLogistic:
    def test_visualize_binary_1d(self, trained_binary_1d):
        model, X, y = trained_binary_1d
        fig = visualize_logistic(model, X, y, steps=5, show_loss=True)
        assert isinstance(fig, go.Figure)
        assert len(fig.frames) == 5

    def test_visualize_binary_2d(self, trained_binary_2d):
        model, X, y = trained_binary_2d
        fig = visualize_logistic(model, X, y, steps=5, show_loss=True)
        assert isinstance(fig, go.Figure)
        assert len(fig.frames) == 5

    def test_feature_names_label_binary_model_input_axes(
        self, trained_binary_1d, trained_binary_2d
    ):
        model_1d, X_1d, y_1d = trained_binary_1d
        fig_1d = visualize_logistic(
            model_1d,
            X_1d,
            y_1d,
            steps=5,
            show_loss=True,
            feature_names=["Ingreso mensual"],
        )
        assert fig_1d.layout.xaxis.title.text == "Ingreso mensual"
        assert fig_1d.layout.xaxis2.title.text == "Replay / fitted state"
        assert fig_1d.layout.yaxis2.title.text == "log-loss"

        model_2d, X_2d, y_2d = trained_binary_2d
        fig_2d = visualize_logistic(
            model_2d,
            X_2d,
            y_2d,
            steps=5,
            feature_names=["Edad", "Ingreso mensual"],
        )
        assert fig_2d.layout.scene.xaxis.title.text == "Edad"
        assert fig_2d.layout.scene.yaxis.title.text == "Ingreso mensual"
        assert fig_2d.layout.scene.zaxis.title.text == "P(class 1 | x)"

    def test_feature_name_labels_multiclass_1d_axis(self, trained_multiclass_1d):
        model, X, y = trained_multiclass_1d
        fig = visualize_logistic(
            model,
            X,
            y,
            steps=5,
            feature_names=["Puntaje"],
        )
        assert fig.layout.xaxis2.title.text == "Puntaje"
        assert float(fig.layout.xaxis2.title.standoff) == pytest.approx(4)

    def test_visualize_binary_nd(self, trained_binary_nd):
        model, X, y = trained_binary_nd
        fig = visualize_logistic(model, X, y, steps=5, show_loss=True)
        assert isinstance(fig, go.Figure)
        assert len(fig.frames) == 5

    def test_binary_nd_wraps_intercept_and_centers_metric_column(self):
        X, y = _make_dummy_classification_data(120, 30, 2)
        model = LogisticRegression(max_iter=1000, random_state=42).fit(X, y)
        fig = visualize_logistic(
            model,
            X,
            y,
            steps=12,
            max_frames=6,
            show_loss=True,
            metrics=["log_loss", "accuracy", "f1"],
            detail="complete",
            format="compact",
            size="wide",
        )

        annotation_texts = [str(annotation.text) for annotation in fig.layout.annotations]
        header = next(text for text in annotation_texts if r"\sum_{j=1}^{30}" in text)
        assert r"\hat{p}(Y=c_1\mid\mathbf{x})=\sigma(z)=\dfrac{1}{1+e^{-z}}" in header

        scalar = next(
            text
            for text in annotation_texts
            if r"\begin{aligned}" in text and r"x_{30}" in text
        )
        assert r"\hat{p}" not in scalar
        assert r"\dfrac" not in scalar
        coefficient_rows = scalar.split(r" \\ ")
        assert r"x_{30}" in coefficient_rows[-2]
        assert r"x_{30}" not in coefficient_rows[-1]
        assert coefficient_rows[-1].startswith(r"&\quad + (")

        metric_annotations = [
            annotation
            for annotation in fig.layout.annotations
            if any(label in str(annotation.text) for label in ("log-loss", "Accuracy", "F1 Score"))
        ]
        assert len(metric_annotations) == 3
        assert len({float(annotation.x) for annotation in metric_annotations}) == 1
        assert {annotation.xanchor for annotation in metric_annotations} == {"center"}
        assert {annotation.align for annotation in metric_annotations} == {"center"}

    def test_visualize_multiclass_1d(self, trained_multiclass_1d):
        model, X, y = trained_multiclass_1d
        fig = visualize_logistic(model, X, y, steps=5, show_loss=True)
        assert isinstance(fig, go.Figure)
        assert len(fig.frames) == 5
        upper_probability_bottom = float(fig.layout.yaxis2.domain[0])
        lower_loss_top = float(fig.layout.yaxis4.domain[1])
        assert upper_probability_bottom - lower_loss_top >= 0.179
        annotations = " ".join(str(item.text) for item in fig.layout.annotations)
        assert r"q_k=\sigma(z_k)" in annotations
        assert r"\Theta^\top\mathbf{x}+\boldsymbol{\theta}_0" in annotations
        assert r"K=3" in annotations
        assert r"\text{clases}" not in annotations
        assert r"k=0,\ldots,2" in annotations
        assert r"\sum_{\ell=0}^{K-1}q_\ell" in annotations
        assert r"\Theta_t" not in annotations
        assert r"\boldsymbol{\theta}_{0,t}" not in annotations

        def math_positions(items):
            row1 = next(item for item in items if r"\mathbf{z}=\Theta^\top" in str(item.text))
            generic = next(item for item in items if r"z_k(\mathbf{x})=" in str(item.text))
            example = next(item for item in items if r"z_0(\mathbf{x})=" in str(item.text))
            ellipsis = next(item for item in items if str(item.text) == r"$$\vdots$$")
            tail = next(item for item in items if r"Y=c_{2}" in str(item.text))
            return row1, generic, example, ellipsis, tail

        for items in [fig.layout.annotations, *(frame.layout.annotations for frame in fig.frames)]:
            row1, generic, example, ellipsis, tail = math_positions(items)
            assert float(row1.x) == pytest.approx(0.33)
            assert generic.text.index(r"\hat{p}_k") < generic.text.index(r"z_k(\mathbf{x})") < generic.text.index("k=0")
            assert r"z_{2}(\mathbf{x})=" in example.text
            assert r"\quad\cdots\quad " in example.text
            assert float(generic.y) > float(example.y) > float(tail.y)
            assert float(ellipsis.x) == pytest.approx(0.328)
            assert float(ellipsis.y) - float(tail.y) >= 0.075
            assert not any(r"\hat p_{0}=" in str(item.text) for item in items)
            assert not any(r"\hat{y}" in str(item.text) for item in items)

    def test_visualize_multiclass_2d(self, trained_multiclass_2d):
        model, X, y = trained_multiclass_2d
        fig = visualize_logistic(model, X, y, steps=5, show_loss=True, detail="complete")
        assert isinstance(fig, go.Figure)
        assert len(fig.frames) == 5

        for items in [fig.layout.annotations, *(frame.layout.annotations for frame in fig.frames)]:
            texts = [str(item.text) for item in items]
            assert not any(r"\text{Para }k=" in text for text in texts)
            equation_rows = [
                item
                for item in items
                if any(
                    marker in str(item.text)
                    for marker in (
                        r"z_k(\mathbf{x})=",
                        r"z_0(\mathbf{x})=",
                        r"Y=c_{0}",
                        r"Y=c_{2}",
                    )
                )
            ]
            assert len(equation_rows) == 4
            assert {item.xanchor for item in equation_rows} == {"center"}
            score_row = next(item for item in equation_rows if r"z_0(\mathbf{x})=" in item.text)
            generic = next(item for item in equation_rows if r"z_k(\mathbf{x})=" in item.text)
            probability_rows = [
                item for item in equation_rows if r"Y=c_{" in str(item.text)
            ]
            assert float(generic.x) == pytest.approx(0.275)
            assert all(float(item.x) == pytest.approx(0.30) for item in (score_row, *probability_rows))
            assert r"\;\cdots\; z_{2}(\mathbf{x})=" in score_row.text
            assert generic.text.index(r"\hat{p}_k") < generic.text.index(r"z_k(\mathbf{x})") < generic.text.index("k=0")
            assert all(
                r"\frac{q_{" in item.text
                and r"\sum_{\ell=0}^{K-1}q_\ell" in item.text
                and item.font.size == 13
                for item in probability_rows
            )
            assert not any(r"\hat p_{0}=" in text or r"\hat{y}" in text for text in texts)

        annotation_text = " ".join(str(item.text) for item in fig.layout.annotations)
        assert r"K=3" in annotation_text
        assert r"\text{clases}" not in annotation_text
        assert r"k=0,\ldots,2" in annotation_text
        assert r"\quadp" not in annotation_text
        assert r"\text{For }c_" in annotation_text

    def test_visualize_multiclass_nd(self, trained_multiclass_nd):
        model, X, y = trained_multiclass_nd
        fig = visualize_logistic(model, X, y, steps=5, show_loss=True)
        assert isinstance(fig, go.Figure)
        assert len(fig.frames) == 5

        for items in [fig.layout.annotations, *(frame.layout.annotations for frame in fig.frames)]:
            texts = [str(item.text) for item in items]
            assert r"K=3" in texts[0]
            assert r"\text{clases}" not in texts[0]
            assert r"\\[5pt]\boldsymbol{\theta}_0&=" in texts[2]
            assert r"\Theta_t" not in texts[2]
            assert texts[3].index(r"\hat{p}_k") < texts[3].index(r"z_k(\mathbf{x})=")
            assert r"\frac{q_k}{\sum_{\ell=0}^{K-1}q_\ell}" in texts[3]
            assert r"\sigma(z_k)" in texts[3]
            assert r"k=0,\ldots,2" in texts[3]
            assert r"z_{0}(\mathbf{x})=" in texts[4]
            assert r"\;\cdots\; z_{2}(\mathbf{x})=" in texts[4]
            assert r"\begin{array}{r@{\,}c@{\,}l}" in texts[5]
            assert r"Y=c_{0}" in texts[5]
            assert r"&&\smash{\overset{\hphantom{" in texts[5]
            assert r"}{\vdots}" in texts[5]
            assert r"Y=c_{2}" in texts[5]
            assert [float(item.y) for item in items[3:6]] == pytest.approx(
                [0.50, 0.345, 0.22]
            )
            assert float(items[0].x) == pytest.approx(0.46)
            assert float(items[1].y) == pytest.approx(0.77)
            assert float(items[2].x) == pytest.approx(0.10)
            assert float(items[2].y) == pytest.approx(0.77)
            assert len({float(item.x) for item in items[3:6]}) == 1
            assert float(items[3].x) == pytest.approx(0.015)
            assert {item.xanchor for item in items[3:6]} == {"left"}
            assert not any(r"\hat p_{0}=" in text or r"\hat{y}" in text for text in texts)

    @pytest.mark.parametrize("size", ["compact", "notebook"])
    @pytest.mark.parametrize("detail", ["essential", "complete"])
    def test_narrow_multiclass_spacing_keeps_continuation_marks_centered(self, size, detail):
        X, y = _make_dummy_classification_data(90, 6, 6)
        model = LogisticRegression(max_iter=1000).fit(X, y)
        fig = visualize_logistic(
            model, X, y, steps=3, show_loss=True, max_theta_cols=2,
            size=size, detail=detail,
        )
        for items in [fig.layout.annotations, *(frame.layout.annotations for frame in fig.frames)]:
            assert "shown columns" not in items[2].text
            assert r" & \cdots & " in items[2].text
            assert [float(item.y) for item in items[3:6]] == pytest.approx(
                [0.48, 0.34, 0.16]
            )
            assert float(items[1].y) == pytest.approx(0.79)
            assert float(items[2].y) == pytest.approx(0.74)
            assert r"\begin{array}{l}" in items[4].text
            assert r"\hphantom{\displaystyle z_" in items[4].text
            assert r"}{\vdots}" in items[4].text
            assert items[5].text.count(r"\\[0pt]") == 2

    @pytest.mark.parametrize("size", ["default", "wide"])
    def test_dense_multiclass_nd_expands_exponents_in_standard_and_wide_size(self, size):
        X, y = _make_dummy_classification_data(120, 13, 3)
        model = LogisticRegression(max_iter=1000, random_state=42).fit(X, y)
        fig = visualize_logistic(
            model,
            X,
            y,
            steps=6,
            show_loss=True,
            metrics=["log_loss", "accuracy", "f1"],
            max_theta_cols=2,
            detail="complete",
            size=size,
        )

        for items in [fig.layout.annotations, *(frame.layout.annotations for frame in fig.frames)]:
            texts = [str(item.text) for item in items]
            assert r"\Theta&=" in texts[2]
            assert r"\Theta_t" not in texts[2]
            assert "shown columns" not in texts[2]
            assert r"\begin{array}{ccc}" in texts[2]
            assert r"\boldsymbol{\theta}_0&=\begin{bmatrix}" in texts[2]
            assert r" & \cdots & " in texts[2]
            assert " & " not in texts[1]
            assert r"x_{13}" in texts[1]
            assert r"\vdots" in texts[1]
            assert r"\hat{p}_k" in texts[3]
            assert r"z_k(\mathbf{x})=" in texts[3]
            assert r"z_{0}(\mathbf{x})=" in texts[4]
            assert r"x_{13}" in texts[4]
            assert r"\;\cdots\; z_{2}(\mathbf{x})=" in texts[4]
            assert texts[5].count(r"\hat{p}(Y=c_{") == 2
            # Eight visible exponent terms plus the four-term phantom used to
            # center the omission mark under the widest denominator.
            assert texts[5].count(r"e^{\left(") == 12
            assert texts[5].count(r"x_{13}") == 12
            assert r"+\cdots+" in texts[5]
            assert r"&&\smash{\overset{\hphantom{" in texts[5]
            assert r"}{\vdots}" in texts[5]
            assert r"\\[6pt]" in texts[5]
            assert r"\displaystyle \frac" in texts[5]
            assert [float(item.y) for item in items[3:6]] == pytest.approx(
                [0.44, 0.285, 0.17]
            )
            assert float(items[0].x) == pytest.approx(0.46)
            assert float(items[2].x) == pytest.approx(0.10)
            assert len({float(item.x) for item in items[3:6]}) == 1
            assert float(items[3].x) == pytest.approx(0.015)
            assert not any(r"\hat p_{0}=" in text or r"\hat{y}" in text for text in texts)

        compact = visualize_logistic(
            model, X, y, steps=6, show_loss=True, max_theta_cols=2,
            detail="complete", size="notebook",
        )
        for items in [compact.layout.annotations, *(frame.layout.annotations for frame in compact.frames)]:
            assert items[5].text.count(r"e^{z_{") == 12
            assert r"&&\smash{\overset{\hphantom{" in items[5].text
            assert r"}{\vdots}" in items[5].text
            assert float(items[5].x) == pytest.approx(0.015)
            assert r"x_{13}" in items[4].text
            assert r"\begin{array}{l}" in items[4].text
            assert r"\hphantom{\displaystyle z_" in items[4].text
            assert "shown columns" not in items[2].text
            assert [float(item.y) for item in items[3:6]] == pytest.approx(
                [0.46, 0.325, 0.12]
            )
            assert items[5].text.count(r"\\[0pt]") == 2

        full_columns = visualize_logistic(
            model, X, y, steps=3, max_theta_cols=3, size="notebook",
        )
        matrix_text = full_columns.layout.annotations[2].text
        assert r"\begin{array}{ccc}" in matrix_text
        assert r"\text{shown columns: }" not in matrix_text
        assert r"\cdots" not in matrix_text

    def test_probability_link_matches_estimator_semantics(self):
        X, y = _make_dummy_classification_data(80, 2, 3)
        estimators = [
            SGDClassifier(loss="log_loss", max_iter=1000, random_state=42),
            LogisticRegression(max_iter=1000, random_state=42),
        ]
        expected_links = ["ovr", "softmax"]
        for estimator, expected_link in zip(estimators, expected_links):
            estimator.fit(X, y)
            history = fit_history_logistic(estimator, X, y, steps=3, mode="final_interp")
            scores = estimator.decision_function(X)
            reconstructed = multiclass_probabilities(scores, history["probability_link"])
            assert history["probability_link"] == expected_link
            assert np.allclose(reconstructed, estimator.predict_proba(X), atol=1e-10)

    def test_animation_transition_is_smooth_for_2d_traces(self, trained_binary_1d):
        model, X, y = trained_binary_1d
        fig = visualize_logistic(
            model,
            X,
            y,
            steps=4,
            frame_duration=100,
            transition_duration=75,
        )
        options = fig.layout.updatemenus[0].buttons[0].args[1]
        assert options["transition"]["duration"] == 75
        assert options["frame"]["redraw"] is True
        assert fig.layout.transition.ordering == "traces first"

    def test_high_dimensional_multiclass_layout_keeps_true_dimensions(self):
        rng = np.random.default_rng(8)
        steps, samples, dimensions, classes = 3, 40, 20, 12
        X = rng.normal(size=(samples, dimensions))
        y = np.arange(samples) % classes
        history = {
            "history_kind": "iterative",
            "classes": np.arange(classes),
            "probability_link": "softmax",
            "w_hist": np.zeros((steps, dimensions, classes)),
            "b_hist": np.zeros((steps, classes)),
            "loss_hist": np.linspace(2.5, 1.5, steps),
        }
        fig = build_logistic_figure(X, y, history=history, show_loss=True, max_theta_cols=5)
        annotations = " ".join(str(item.text) for item in fig.layout.annotations)
        assert rf"\mathbb{{R}}^{{{dimensions}\times {classes}}}" in annotations
        assert r"\cdots" in annotations
        assert len(fig.frames) == steps
        for annotation in fig.layout.annotations:
            if annotation.xref == "paper":
                assert -0.15 <= float(annotation.x) <= 1.05
                assert -0.15 <= float(annotation.y) <= 1.15

    def test_visualize_pipeline_scaled(self, trained_pipeline_binary_1d):
        model, X, y = trained_pipeline_binary_1d
        fig = visualize_logistic(model, X, y, steps=5, display_space="scaled")
        assert isinstance(fig, go.Figure)
        assert len(fig.frames) == 5

    def test_visualize_pipeline_original(self, trained_pipeline_binary_1d):
        model, X, y = trained_pipeline_binary_1d
        fig = visualize_logistic(model, X, y, steps=5, display_space="original")
        assert isinstance(fig, go.Figure)
        assert len(fig.frames) == 5

class TestExplainLogisticPrediction:
    @pytest.mark.parametrize("size", ["default", "compact", "notebook", "wide"])
    def test_binary_1d_gives_substitution_more_height_than_single_input(self, trained_binary_1d, size):
        model, X, y = trained_binary_1d
        fig = explain_logistic_prediction(model, X, y, x_query=X[0], size=size)
        inputs, substitution, result = fig.layout.shapes
        assert substitution.y1 - substitution.y0 > 2 * (inputs.y1 - inputs.y0)
        assert result.y1 < substitution.y0 < substitution.y1 < inputs.y0
        for stage in fig.layout.updatemenus[0].buttons[:3]:
            annotation = stage.args[1]["annotations"][3]
            assert substitution.y0 < annotation["y"] < substitution.y1
        assert fig.layout.updatemenus[0].buttons[1].args[1]["annotations"][3]["y"] == pytest.approx(0.68)

    def test_explain_prediction_binary_1d(self, trained_binary_1d):
        model, X, y = trained_binary_1d
        fig = explain_logistic_prediction(model, X, y, x_query=X[0])
        assert isinstance(fig, go.Figure)
        assert "Prediction" in fig.layout.title.text
        assert tuple(fig.layout.yaxis2.ticktext) == ("0", "0.5", "1")
        assert all("threshold" not in str(value).lower() for value in fig.layout.yaxis2.ticktext)

    def test_explain_prediction_pipeline_scaled(self, trained_pipeline_binary_1d):
        model, X, y = trained_pipeline_binary_1d
        fig = explain_logistic_prediction(model, X, y, x_query=X[0], display_space="scaled")
        assert isinstance(fig, go.Figure)

    def test_explain_prediction_pipeline_original(self, trained_pipeline_binary_1d):
        model, X, y = trained_pipeline_binary_1d
        fig = explain_logistic_prediction(model, X, y, x_query=X[0], display_space="original")
        assert isinstance(fig, go.Figure)

    def test_explain_prediction_binary_2d(self, trained_binary_2d):
        model, X, y = trained_binary_2d
        fig = explain_logistic_prediction(model, X, y, x_query=X[0])
        assert isinstance(fig, go.Figure)
        assert "Prediction" in fig.layout.title.text

    def test_explain_prediction_multiclass_1d(self):
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        X = np.linspace(-5, 5, 100).reshape(-1, 1)
        y = np.zeros(100)
        y[X.ravel() > -1] = 1
        y[X.ravel() > 2] = 2
        model = LogisticRegression(random_state=42)
        model.fit(X, y)
        fig = explain_logistic_prediction(model, X, y, x_query=X[0])
        assert isinstance(fig, go.Figure)
        assert "Multiclass" in fig.layout.title.text
