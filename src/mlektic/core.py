"""
Core module for the mlektic library.

This module provides the main API for extracting training history from scikit-learn
models and visualizing their parameters and predictions over time.
"""

import numpy as np
from sklearn.base import clone
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline


# -------------------------
# Helpers
# -------------------------
def _first_not_none(*args):
    """Return the first non-None argument from the given list of arguments."""
    for a in args:
        if a is not None:
            return a
    return None


def _as_2d(X):
    """Ensure that the input array X is 2-dimensional."""
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return X


def _as_1d(y):
    """Ensure that the input array y is 1-dimensional."""
    return np.asarray(y).ravel()


def _get_final_estimator(estimator):
    """Extract the final estimator from a Pipeline, or return the estimator itself."""
    if isinstance(estimator, Pipeline):
        return estimator.steps[-1][1]
    return estimator


def _last_step_prefix(estimator):
    """Extract the string prefix of the last step in a Pipeline, or None."""
    if isinstance(estimator, Pipeline):
        return estimator.steps[-1][0]
    return None


def _try_set_params(estimator, **params):
    """Safely attempt to set parameters on an estimator, catching any exceptions."""
    try:
        estimator.set_params(**params)
        return True
    except Exception:
        return False


def _ema_smooth(arr, beta=0.85):
    """Apply Exponential Moving Average smoothing to a 1D array."""
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        return arr
    out = np.empty_like(arr, dtype=float)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = beta * out[i - 1] + (1 - beta) * arr[i]
    return out


def _is_iterative(estimator):
    """Check if the estimator supports iterative training (partial_fit or warm_start)."""
    last = _get_final_estimator(estimator)
    return hasattr(last, "partial_fit") or hasattr(last, "warm_start")


def _extract_theta_as_learned(estimator, d_expected=None):
    """
    Devuelve (w, b) tal cual están en el estimador FINAL (último step si Pipeline)
    después del fit(). NO destransforma.

    Si d_expected se da y no coincide con coef_, regresa (None, None).
    """
    est = _get_final_estimator(estimator)

    if not (hasattr(est, "coef_") and hasattr(est, "intercept_")):
        return None, None

    w = np.asarray(est.coef_, dtype=float).ravel()

    b_raw = np.asarray(est.intercept_, dtype=float).ravel()
    b = float(b_raw[0]) if b_raw.size else float(est.intercept_)

    if d_expected is not None and w.size != int(d_expected):
        return None, None

    return w, b


def _find_standard_scaler(estimator):
    """
    Search for a StandardScaler-like step inside a Pipeline estimator.

    Checks if there is a step with `mean_`, `scale_` (or `var_`), and `transform`.
    """
    if not isinstance(estimator, Pipeline):
        return None

    for _, step in estimator.steps:
        has_transform = hasattr(step, "transform")
        has_mean = hasattr(step, "mean_")
        has_scale = hasattr(step, "scale_") or hasattr(step, "var_")
        if has_transform and has_mean and has_scale:
            return step
    return None


def _safe_get_scale(scaler):
    """Extract mean and scale properties safely from a scaler object."""
    if scaler is None:
        return None, None, True, True

    mu = getattr(scaler, "mean_", None)

    scale = getattr(scaler, "scale_", None)
    if scale is None:
        var = getattr(scaler, "var_", None)
        if var is not None:
            scale = np.sqrt(np.asarray(var, dtype=float))
        else:
            scale = None

    with_mean = bool(getattr(scaler, "with_mean", True))
    with_std = bool(getattr(scaler, "with_std", True))

    return mu, scale, with_mean, with_std


def _theta_scaled_to_original(w_s, b_s, scaler):
    """
    Convierte (w,b) del espacio escalado al original (StandardScaler en X):

    y = b_s + w_s * ((x - mu)/scale)
    => w_o = w_s/scale
       b_o = b_s - sum_j w_sj*mu_j/scale_j
    """
    w_s = np.asarray(w_s, dtype=float).ravel()
    b_s = float(np.asarray(b_s, dtype=float).ravel()[0]) if np.size(b_s) else float(b_s)

    if scaler is None:
        return w_s.copy(), float(b_s)

    mu, scale, with_mean, with_std = _safe_get_scale(scaler)
    dloc = w_s.size

    if (not with_std) or (scale is None):
        scale = np.ones(dloc, dtype=float)
    else:
        scale = np.asarray(scale, dtype=float).ravel()
        if scale.size != dloc:
            raise ValueError(f"Scaler/coef mismatch: scale has {scale.size}, coef has {dloc}.")

    if (not with_mean) or (mu is None):
        mu = np.zeros(dloc, dtype=float)
    else:
        mu = np.asarray(mu, dtype=float).ravel()
        if mu.size != dloc:
            raise ValueError(f"Scaler/coef mismatch: mean has {mu.size}, coef has {dloc}.")

    denom = scale + 1e-12
    w_o = w_s / denom
    b_o = float(b_s - np.sum(w_s * mu / denom))
    return w_o, b_o


def _transform_up_to_last(pipeline, X):
    """
    Apply all steps of a Pipeline EXCEPT the final estimator.

    Returns the transformed input matrix X_transformed.
    """
    Xt = X
    for _, step in pipeline.steps[:-1]:
        if hasattr(step, "transform"):
            Xt = step.transform(Xt)
    return Xt


def _make_iterative_replay_estimator(estimator):
    """
    Clone the estimator and try to force it into iterative training mode.

    Sets warm_start=True, max_iter=1, tol=None, and shuffle=False on a
    best-effort basis.
    """
    est = clone(estimator)

    pref = _last_step_prefix(est)

    def p(name):
        return f"{pref}__{name}" if pref is not None else name

    _try_set_params(est, **{p("warm_start"): True})
    _try_set_params(est, **{p("max_iter"): 1})
    _try_set_params(est, **{p("tol"): None})
    _try_set_params(est, **{p("shuffle"): False})
    return est


# -------------------------
# Main: capture predictive history (robust)
# -------------------------
def fit_history(
    trained_estimator,
    X,
    y,
    *,
    steps=60,
    mode="auto",  # "auto" | "iterative" | "final_interp"
    smooth=None,  # None | "ema"
    smooth_beta=0.85,
    grid_1d_points=250,
    grid_2d_points=40,
    baseline="mean",  # "mean" | "zeros"
    display_space="original",  # "original" | "scaled" (solo afecta theta display)
):
    """
    Historia robusta para casi cualquier sklearn estimator / Pipeline:

    A) iterative:
       - Reproduce el entrenamiento sobre un clone (best-effort)
       - Captura predicciones por frame SIEMPRE usando predict()
       - Captura (w,b) si el estimador final expone coef_ e intercept_

    B) final_interp:
       - Interpola PREDICCIONES (no parámetros) entre un baseline y el modelo final
       - Es agnóstico a transforms / pipelines

    display_space:
      - "scaled":   w_hist/b_hist tal cual aprendidos por el estimador final
      - "original": si hay scaler tipo StandardScaler en el Pipeline, convierte w_hist/b_hist a espacio original
                    (solo para mostrar en animación; no afecta predicciones ni loss)

    Returns dict:
      history_kind: "iterative" | "final_interp"
      loss_hist: (steps,)
      grid: dict
      y_line_hist / z_plane_hist: historia de predicciones en el espacio ORIGINAL de entrada
      w_hist / b_hist: theta "para mostrar" según display_space (si disponible)
      w_hist_learned / b_hist_learned: theta tal cual aprendida (si disponible)
    """
    if steps < 1:
        raise ValueError("steps must be >= 1.")
    if display_space not in ("original", "scaled"):
        raise ValueError("display_space must be 'original' or 'scaled'.")
    if baseline not in ("mean", "zeros"):
        raise ValueError("baseline must be 'mean' or 'zeros'.")
    if mode not in ("auto", "iterative", "final_interp"):
        raise ValueError("mode must be 'auto', 'iterative', or 'final_interp'.")

    X = _as_2d(X)
    y = _as_1d(y)
    n, d = X.shape

    # --- scaler (si existe dentro de Pipeline) ---
    # Nota: en modo iterative, recalcularemos el scaler desde el replay ya fit,
    # para que la conversión de theta sea consistente con ese entrenamiento.
    scaler_trained = _find_standard_scaler(trained_estimator)

    # build grids for plotting (only if d=1 or d=2)
    grid = {}
    x1_grid = None
    X1g = X2g = None

    if d == 1:
        x1 = X[:, 0]
        x_min, x_max = float(x1.min()), float(x1.max())
        x1_grid = np.linspace(x_min, x_max, int(grid_1d_points))
        grid["x1_grid"] = x1_grid

    elif d == 2:
        x1 = X[:, 0]
        x2 = X[:, 1]
        x1_grid = np.linspace(float(x1.min()), float(x1.max()), int(grid_2d_points))
        x2_grid = np.linspace(float(x2.min()), float(x2.max()), int(grid_2d_points))
        X1g, X2g = np.meshgrid(x1_grid, x2_grid)
        grid["x1_grid"] = x1_grid
        grid["x2_grid"] = x2_grid
        grid["X1g"] = X1g
        grid["X2g"] = X2g

    # decide mode
    iterative_like = _is_iterative(trained_estimator)
    if mode == "auto":
        mode = "iterative" if iterative_like else "final_interp"

    loss_hist = np.zeros(steps, dtype=float)

    w_hist = None
    b_hist = None

    # allocate pred grids history
    y_line_hist = None
    z_plane_hist = None
    if d == 1 and x1_grid is not None:
        y_line_hist = np.zeros((steps, x1_grid.size), dtype=float)
    if d == 2 and X1g is not None:
        z_plane_hist = np.zeros((steps, X1g.shape[0], X1g.shape[1]), dtype=float)

    def _pred_grid(est):
        if d == 1 and x1_grid is not None:
            Xg = x1_grid.reshape(-1, 1)
            return est.predict(Xg)  # (G,)
        if d == 2 and X1g is not None:
            Xg = np.column_stack([X1g.ravel(), X2g.ravel()])
            z = est.predict(Xg).reshape(X1g.shape[0], X1g.shape[1])
            return z
        return None

    history_kind = None
    scaler_for_display = scaler_trained  # puede ser reemplazado en iterative

    # ============================================================
    # A) ITERATIVE: replay (best-effort)
    # ============================================================
    if mode == "iterative":
        if not iterative_like:
            mode = "final_interp"
        else:
            est_replay = _make_iterative_replay_estimator(trained_estimator)

            # init internal state (fit 1 vez)
            est_replay.fit(X, y)

            # para display_space="original", usa el scaler del replay ya fitteado (si existe)
            scaler_for_display = _find_standard_scaler(est_replay)

            # init theta hist if available (AS LEARNED)
            w0_t, b0_t = _extract_theta_as_learned(est_replay, d_expected=d)
            if w0_t is not None:
                w_hist = np.zeros((steps, d), dtype=float)
                b_hist = np.zeros(steps, dtype=float)
                w_hist[0] = w0_t
                b_hist[0] = b0_t

            # reset SGD schedule counter if present (best effort)
            last = _get_final_estimator(est_replay)
            if hasattr(last, "t_"):
                try:
                    last.t_ = 1.0
                except Exception:
                    pass

            # step 0 (ALWAYS from predict)
            y_pred0 = est_replay.predict(X)
            loss_hist[0] = float(mean_squared_error(y, y_pred0))

            g0 = _pred_grid(est_replay)
            if d == 1 and y_line_hist is not None:
                y_line_hist[0] = np.asarray(g0, dtype=float)
            if d == 2 and z_plane_hist is not None:
                z_plane_hist[0] = np.asarray(g0, dtype=float)

            # steps 1..T-1
            is_pipe = isinstance(est_replay, Pipeline)
            last_step_est = _get_final_estimator(est_replay)
            has_pf_last = hasattr(last_step_est, "partial_fit")

            can_pf_pipe = False
            Xt = None
            if is_pipe:
                # Pipeline: para partial_fit solo del último step necesitamos X transformado
                try:
                    Xt = _transform_up_to_last(est_replay, X)
                    can_pf_pipe = hasattr(est_replay.steps[-1][1], "partial_fit")
                except Exception:
                    Xt = None
                    can_pf_pipe = False

            for t in range(1, steps):
                if (not is_pipe) and has_pf_last:
                    # Estimador directo con partial_fit
                    est_replay.partial_fit(X, y)

                elif is_pipe and can_pf_pipe and Xt is not None:
                    # Pipeline: actualiza SOLO el último step con partial_fit
                    # (evita re-entrenar el scaler por frame)
                    est_replay.steps[-1][1].partial_fit(Xt, y)

                else:
                    # Fallback: fit completo (best-effort)
                    est_replay.fit(X, y)

                y_pred = est_replay.predict(X)
                loss_hist[t] = float(mean_squared_error(y, y_pred))

                gt = _pred_grid(est_replay)
                if d == 1 and y_line_hist is not None:
                    y_line_hist[t] = np.asarray(gt, dtype=float)
                if d == 2 and z_plane_hist is not None:
                    z_plane_hist[t] = np.asarray(gt, dtype=float)

                if w_hist is not None:
                    wt, bt = _extract_theta_as_learned(est_replay, d_expected=d)
                    if wt is not None:
                        w_hist[t] = wt
                        b_hist[t] = bt

            history_kind = "iterative"

    # ============================================================
    # B) FINAL_INTERP: interpolate *predictions* (robust to transforms)
    # ============================================================
    if mode == "final_interp":
        # baseline predictions
        if baseline == "zeros":
            y0 = np.zeros_like(y, dtype=float)
        else:
            y0 = np.full_like(y, float(np.mean(y)), dtype=float)

        # final predictions from trained estimator
        yF = np.asarray(trained_estimator.predict(X), dtype=float)

        # baseline grid preds
        g0 = None
        gF = None

        if d == 1 and y_line_hist is not None:
            if baseline == "zeros":
                g0 = np.zeros_like(y_line_hist[0])
            else:
                g0 = np.full_like(y_line_hist[0], float(np.mean(y)))
            gF = np.asarray(
                trained_estimator.predict(grid["x1_grid"].reshape(-1, 1)),
                dtype=float,
            )

        if d == 2 and z_plane_hist is not None:
            if baseline == "zeros":
                g0 = np.zeros_like(z_plane_hist[0])
            else:
                g0 = np.full_like(z_plane_hist[0], float(np.mean(y)))
            Xg = np.column_stack([grid["X1g"].ravel(), grid["X2g"].ravel()])
            gF = np.asarray(trained_estimator.predict(Xg), dtype=float).reshape(grid["X1g"].shape)

        # theta final (AS LEARNED) si coincide con d
        wF, bF = _extract_theta_as_learned(trained_estimator, d_expected=d)
        if wF is not None:
            w_hist = np.tile(wF.reshape(1, -1), (steps, 1))
            b_hist = np.full(steps, float(bF), dtype=float)

        for t in range(steps):
            alpha = t / (steps - 1) if steps > 1 else 1.0
            y_pred = (1 - alpha) * y0 + alpha * yF
            loss_hist[t] = float(mean_squared_error(y, y_pred))

            if d == 1 and y_line_hist is not None and g0 is not None:
                y_line_hist[t] = (1 - alpha) * g0 + alpha * gF
            if d == 2 and z_plane_hist is not None and g0 is not None:
                z_plane_hist[t] = (1 - alpha) * g0 + alpha * gF

        history_kind = "final_interp"

    # -------------------------
    # Optional smoothing (VISUAL ONLY)
    # -------------------------
    if smooth == "ema":
        loss_hist = _ema_smooth(loss_hist, beta=smooth_beta)

        if y_line_hist is not None:
            # suaviza cada columna (gridpoint) a través del tiempo
            for j in range(y_line_hist.shape[1]):
                y_line_hist[:, j] = _ema_smooth(y_line_hist[:, j], beta=smooth_beta)

        if z_plane_hist is not None:
            Z = z_plane_hist.reshape(steps, -1)
            for j in range(Z.shape[1]):
                Z[:, j] = _ema_smooth(Z[:, j], beta=smooth_beta)
            z_plane_hist = Z.reshape(z_plane_hist.shape)

    # -------------------------
    # theta "para mostrar" según display_space (sin afectar pred grids)
    # -------------------------
    w_hist_learned = None
    b_hist_learned = None

    if w_hist is not None and b_hist is not None:
        # guardar learned tal cual
        w_hist_learned = np.asarray(w_hist, dtype=float).copy()
        b_hist_learned = np.asarray(b_hist, dtype=float).copy()

        # si display original y hay scaler, convertir SOLO para mostrar
        if display_space == "original" and scaler_for_display is not None:
            w_show = np.zeros_like(w_hist_learned)
            b_show = np.zeros_like(b_hist_learned)
            for t in range(w_hist_learned.shape[0]):
                wo, bo = _theta_scaled_to_original(w_hist_learned[t], b_hist_learned[t], scaler_for_display)
                w_show[t] = wo
                b_show[t] = bo
            w_hist = w_show
            b_hist = b_show
        # else: "scaled" => deja learned tal cual

    return {
        "history_kind": history_kind,
        "loss_hist": loss_hist,
        "grid": grid,
        "y_line_hist": y_line_hist,
        "z_plane_hist": z_plane_hist,
        "w_hist": w_hist,  # theta para mostrar
        "b_hist": b_hist,
        "w_hist_learned": w_hist_learned,  # theta tal cual aprendido (opcional)
        "b_hist_learned": b_hist_learned,
        "display_space": display_space,
    }


import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

pio.renderers.default = "colab"


def build_plane_lr_figure(
    x1,
    x2,
    y,
    w_hist=None,
    b_hist=None,
    *,
    # --- robust inputs (preferred) ---
    z_plane_hist=None,  # (T, H, W) predictions over grid
    X1g=None,
    X2g=None,  # (H, W) meshgrid used to build z_plane_hist
    # --- loss ---
    loss_hist=None,
    show_loss=False,
    history_kind="iterative",
    title="Linear Regression (2 variables)",
    strict_loss=False,
    dec=4,
):
    """
    2D (plane) LR visualization.

    Robust mode (preferred):
      - Provide z_plane_hist + (X1g, X2g) => plot uses predictions, works with ANY sklearn Pipeline/transform.

    Legacy mode:
      - Provide w_hist, b_hist => plot uses z = w1*x1 + w2*x2 + b (only correct for pure linear model in original space)

    Notes:
      - If z_plane_hist is given, w_hist/b_hist are OPTIONAL and only used for display in the equation text.
      - show_loss is only allowed for iterative histories (same rule as 1D).
    """
    # --- enforce inside the library ---
    if show_loss and history_kind != "iterative":
        if strict_loss:
            raise ValueError("show_loss=True is only allowed for iterative histories.")
        show_loss = False
        loss_hist = None

    x1 = np.asarray(x1).ravel()
    x2 = np.asarray(x2).ravel()
    y = np.asarray(y).ravel()

    use_pred_grid = z_plane_hist is not None

    # -------------------------
    # Mode A: robust prediction-grid
    # -------------------------
    if use_pred_grid:
        z_plane_hist = np.asarray(z_plane_hist, dtype=float)

        if X1g is None or X2g is None:
            raise ValueError("If z_plane_hist is provided, X1g and X2g must be provided.")

        X1g = np.asarray(X1g, dtype=float)
        X2g = np.asarray(X2g, dtype=float)

        if X1g.shape != X2g.shape:
            raise ValueError("X1g and X2g must have the same shape.")
        if z_plane_hist.ndim != 3:
            raise ValueError("z_plane_hist must have shape (steps, H, W).")
        if z_plane_hist.shape[1:] != X1g.shape:
            raise ValueError("z_plane_hist grid shape must match X1g/X2g shape.")

        steps_n = int(z_plane_hist.shape[0])
        if steps_n < 1:
            raise ValueError("Need at least 1 step to animate.")

        def z_plane(t: int):
            return z_plane_hist[t]

        def theta_formula_text():
            return r"$\hat{y}=\theta_0+\theta_1 x_1+\theta_2 x_2$"

        # Optional theta display if consistent (accept (T,2) only)
        w_disp = None
        b_disp = None
        if w_hist is not None and b_hist is not None:
            w_arr = np.asarray(w_hist, dtype=float)
            b_arr = np.asarray(b_hist, dtype=float).ravel()
            if w_arr.ndim == 2 and w_arr.shape == (steps_n, 2) and b_arr.size == steps_n:
                w_disp = w_arr
                b_disp = b_arr

        def eq_text(t: int):
            if w_disp is None:
                return r"$\hat{y} = f(x_1,x_2)$"
            w1 = float(w_disp[t, 0])
            w2 = float(w_disp[t, 1])
            b = float(b_disp[t])
            return rf"$\hat{{y}} = ({w1:.{dec}f})x_1 + ({w2:.{dec}f})x_2 + ({b:.{dec}f})$"

        # Ranges driven by GRID (stable + matches surface)
        x1_min, x1_max = float(np.min(X1g)), float(np.max(X1g))
        x2_min, x2_max = float(np.min(X2g)), float(np.max(X2g))

    # -------------------------
    # Mode B: legacy parameter-plane
    # -------------------------
    else:
        if w_hist is None or b_hist is None:
            raise ValueError("Legacy mode requires w_hist and b_hist. Prefer providing z_plane_hist + X1g/X2g.")

        w_hist = np.asarray(w_hist, dtype=float)
        b_hist = np.asarray(b_hist, dtype=float).ravel()
        steps_n = int(b_hist.size)

        if steps_n < 1:
            raise ValueError("Need at least 1 step to animate.")

        # allow w_hist shape flexibility: (T,2) or (T*2,) (rare)
        if w_hist.ndim == 1:
            if w_hist.size == steps_n * 2:
                w_hist = w_hist.reshape(steps_n, 2)
            else:
                raise ValueError("Legacy plane expects w_hist shape (steps, 2) (or flat of length steps*2).")

        if w_hist.ndim != 2 or w_hist.shape != (steps_n, 2):
            raise ValueError("Legacy plane expects w_hist shape (steps, 2) and b_hist shape (steps,)")

        # Build default mesh
        x1_grid = np.linspace(float(x1.min()), float(x1.max()), 40)
        x2_grid = np.linspace(float(x2.min()), float(x2.max()), 40)
        X1g, X2g = np.meshgrid(x1_grid, x2_grid)

        def z_plane(t: int):
            w1 = float(w_hist[t, 0])
            w2 = float(w_hist[t, 1])
            b = float(b_hist[t])
            return w1 * X1g + w2 * X2g + b

        def theta_formula_text():
            return r"$\hat{y}=\theta_0+\theta_1 x_1+\theta_2 x_2$"

        def eq_text(t: int):
            w1 = float(w_hist[t, 0])
            w2 = float(w_hist[t, 1])
            b = float(b_hist[t])
            return rf"$\hat{{y}} = ({w1:.{dec}f})x_1 + ({w2:.{dec}f})x_2 + ({b:.{dec}f})$"

        x1_min, x1_max = float(np.min(X1g)), float(np.max(X1g))
        x2_min, x2_max = float(np.min(X2g)), float(np.max(X2g))

    # -------------------------
    # Validate loss
    # -------------------------
    if show_loss:
        if loss_hist is None:
            raise ValueError("show_loss=True requires loss_hist.")
        loss_hist = np.asarray(loss_hist, dtype=float).ravel()
        if loss_hist.size != steps_n:
            raise ValueError("loss_hist must have the same length as steps.")

    step_axis = np.arange(steps_n)

    # -------------------------
    # Annotations (paper coords)
    # -------------------------
    if show_loss:
        theta_y = 1.16
        eq_y = 1.08
        margin_t = 150
    else:
        theta_y = 1.15
        eq_y = 1.05
        margin_t = 150

    def theta_formula_annotation():
        return dict(
            x=0.5,
            y=theta_y,
            xref="paper",
            yref="paper",
            text=theta_formula_text(),
            showarrow=False,
            xanchor="center",
            yanchor="bottom",
            font=dict(color="white", size=16),
        )

    def eq_annotation(t):
        return dict(
            x=0.5,
            y=eq_y,
            xref="paper",
            yref="paper",
            text=eq_text(t),
            showarrow=False,
            xanchor="center",
            yanchor="bottom",
            font=dict(color="white", size=16),
        )

    # -------------------------
    # Stable scene ranges (use data + plane endpoints)
    # -------------------------
    z0 = np.asarray(z_plane(0), dtype=float)
    zL = np.asarray(z_plane(steps_n - 1), dtype=float)
    z_all = np.concatenate([y, z0.ravel(), zL.ravel()])
    z_min, z_max = float(z_all.min()), float(z_all.max())

    def _pad(lo, hi, frac=0.10):
        span = (hi - lo) + 1e-9
        return [lo - frac * span, hi + frac * span]

    x1_range = _pad(x1_min, x1_max)
    x2_range = _pad(x2_min, x2_max)
    y_range = _pad(z_min, z_max)

    CAMERA = dict(eye=dict(x=1.55, y=1.55, z=1.15))

    if show_loss:
        lmin, lmax = float(loss_hist.min()), float(loss_hist.max())
        lpad = 0.10 * (lmax - lmin + 1e-9)

    # -------------------------
    # Build figure
    # -------------------------
    if show_loss:
        fig = make_subplots(
            rows=1,
            cols=2,
            column_widths=[0.60, 0.30],
            horizontal_spacing=0.06,
            specs=[[{"type": "scene"}, {"type": "xy"}]],
        )

        fig.add_trace(
            go.Scatter3d(
                x=x1,
                y=x2,
                z=y,
                mode="markers",
                name="Data",
                marker=dict(size=4, opacity=0.85),
                legendgroup="fit",
                showlegend=True,
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Surface(
                x=X1g,
                y=X2g,
                z=z_plane(0),
                name="Model",
                opacity=0.55,
                showscale=False,
                legendgroup="fit",
                showlegend=True,
                uid="MODEL_PLANE",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=[0],
                y=[loss_hist[0]],
                mode="lines",
                name="Loss",
                line=dict(width=3),  # don't hardcode colors
                legendgroup="loss",
                showlegend=True,
                uid="LOSS_LINE",
            ),
            row=1,
            col=2,
        )

        frames = []
        for t in range(steps_n):
            frames.append(
                go.Frame(
                    name=str(t),
                    data=[
                        go.Surface(
                            x=X1g,
                            y=X2g,
                            z=z_plane(t),
                            opacity=0.55,
                            showscale=False,
                            showlegend=True,
                            uid="MODEL_PLANE",
                        ),
                        go.Scatter(
                            x=step_axis[: t + 1],
                            y=loss_hist[: t + 1],
                            mode="lines",
                            line=dict(width=3),
                            uid="LOSS_LINE",
                        ),
                    ],
                    traces=[1, 2],
                    layout=go.Layout(
                        annotations=[theta_formula_annotation(), eq_annotation(t)],
                        scene=dict(camera=CAMERA),
                    ),
                )
            )
        fig.frames = frames

        fig.update_layout(
            template="plotly_dark",
            font=dict(family="Helvetica", color="white"),
            height=720,
            title=dict(
                text=title,
                x=0.5,
                y=0.96,
                xanchor="center",
                font=dict(color="white", size=24),
            ),
            annotations=[theta_formula_annotation(), eq_annotation(0)],
            margin=dict(t=margin_t, r=50, l=60, b=70),
            legend=dict(
                x=0.585,
                y=0.82,
                xanchor="right",
                yanchor="bottom",
                bgcolor="rgba(220,220,220,0.85)",
                bordercolor="rgba(0,0,0,0.6)",
                borderwidth=1,
                font=dict(color="black", size=12),
            ),
            legend2=dict(
                x=0.995,
                y=0.82,
                xanchor="right",
                yanchor="bottom",
                bgcolor="rgba(220,220,220,0.85)",
                bordercolor="rgba(0,0,0,0.6)",
                borderwidth=1,
                font=dict(color="black", size=12),
            ),
            scene=dict(
                xaxis=dict(title="x₁", range=x1_range),
                yaxis=dict(title="x₂", range=x2_range),
                zaxis=dict(title="y", range=y_range),
                aspectmode="cube",
                camera=CAMERA,
            ),
            sliders=[
                dict(
                    active=0,
                    currentvalue=dict(prefix="Step: "),
                    pad=dict(t=55),
                    steps=[
                        dict(
                            method="animate",
                            args=[
                                [str(t)],
                                {
                                    "mode": "immediate",
                                    "frame": {"duration": 0, "redraw": True},
                                    "transition": {"duration": 0},
                                },
                            ],
                            label=str(t),
                        )
                        for t in range(steps_n)
                    ],
                )
            ],
            updatemenus=[
                dict(
                    type="buttons",
                    direction="left",
                    x=0.10,
                    y=1.14,
                    bgcolor="white",
                    bordercolor="black",
                    borderwidth=1,
                    font=dict(color="black", size=14),
                    buttons=[
                        dict(
                            label="Play",
                            method="animate",
                            args=[None, {"frame": {"duration": 80, "redraw": True}, "transition": {"duration": 0}}],
                        ),
                        dict(
                            label="Pause",
                            method="animate",
                            args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
                        ),
                    ],
                )
            ],
        )

        # put loss trace in legend2
        fig.data[2].update(legend="legend2")

        fig.update_xaxes(title="Step", range=[0, steps_n - 1], row=1, col=2)
        fig.update_yaxes(title="Loss", range=[lmin - lpad, lmax + lpad], row=1, col=2)

        return fig

    # -------------------------
    # Without loss: single 3D plane
    # -------------------------
    fig = go.Figure()

    fig.add_trace(
        go.Scatter3d(
            x=x1,
            y=x2,
            z=y,
            mode="markers",
            name="Data",
            marker=dict(size=4, opacity=0.85),
        )
    )

    fig.add_trace(
        go.Surface(
            x=X1g,
            y=X2g,
            z=z_plane(0),
            name="Model",
            opacity=0.55,
            showscale=False,
            showlegend=True,
            uid="MODEL_PLANE",
        )
    )

    frames = []
    for t in range(steps_n):
        frames.append(
            go.Frame(
                name=str(t),
                data=[
                    go.Surface(
                        x=X1g,
                        y=X2g,
                        z=z_plane(t),
                        opacity=0.55,
                        showscale=False,
                        showlegend=True,
                        uid="MODEL_PLANE",
                    )
                ],
                traces=[1],
                layout=go.Layout(
                    annotations=[theta_formula_annotation(), eq_annotation(t)],
                    scene=dict(camera=CAMERA),
                ),
            )
        )
    fig.frames = frames

    fig.update_layout(
        template="plotly_dark",
        height=720,
        font=dict(family="Helvetica", color="white"),
        title=dict(
            text=title,
            y=0.96,
            x=0.5,
            xanchor="center",
            font=dict(color="white", size=24),
        ),
        annotations=[theta_formula_annotation(), eq_annotation(0)],
        margin=dict(t=margin_t, r=30, l=60, b=70),
        showlegend=True,
        legend=dict(
            x=0.985,
            y=0.02,
            xanchor="right",
            yanchor="bottom",
            bgcolor="rgba(220,220,220,0.85)",
            bordercolor="rgba(0,0,0,0.6)",
            borderwidth=1,
            font=dict(color="black", size=12),
        ),
        scene=dict(
            xaxis=dict(title="x₁", range=x1_range),
            yaxis=dict(title="x₂", range=x2_range),
            zaxis=dict(title="y", range=y_range),
            aspectmode="cube",
            camera=CAMERA,
        ),
        sliders=[
            dict(
                active=0,
                currentvalue=dict(prefix="Step: "),
                pad=dict(t=55),
                steps=[
                    dict(
                        method="animate",
                        args=[
                            [str(t)],
                            {
                                "mode": "immediate",
                                "frame": {"duration": 0, "redraw": True},
                                "transition": {"duration": 0},
                            },
                        ],
                        label=str(t),
                    )
                    for t in range(steps_n)
                ],
            )
        ],
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                x=0.10,
                y=1.14,
                bgcolor="white",
                bordercolor="black",
                borderwidth=1,
                font=dict(color="black", size=14),
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[None, {"frame": {"duration": 80, "redraw": True}, "transition": {"duration": 0}}],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
                    ),
                ],
            )
        ],
    )

    return fig


import plotly.io as pio

pio.renderers.default = "colab"


def build_simple_lr_figure(
    x1,
    y,
    w_hist=None,
    b_hist=None,
    *,
    # --- robust inputs (preferred) ---
    y_line_hist=None,  # (T, G)
    x1_grid=None,  # (G,)
    # --- loss ---
    loss_hist=None,
    show_loss=False,
    history_kind="iterative",
    title="Linear Regression (Simple, 1 variable)",
    strict_loss=False,
    dec=4,
):
    """
    Simple (1D) visualization.

    Robust mode:
      - Provide y_line_hist + x1_grid => plot uses predictions, works with ANY sklearn Pipeline/transform.

    Legacy mode:
      - Provide w_hist,b_hist => plot uses y = w*x + b (only correct for pure linear model in original space)
    """
    # --- enforce inside the library ---
    if show_loss and history_kind != "iterative":
        if strict_loss:
            raise ValueError("show_loss=True is only allowed for iterative histories.")
        show_loss = False
        loss_hist = None

    x1 = np.asarray(x1).ravel()
    y = np.asarray(y).ravel()

    use_pred_grid = y_line_hist is not None

    # -------------------------
    # Select mode + validate inputs
    # -------------------------
    if use_pred_grid:
        y_line_hist = np.asarray(y_line_hist, dtype=float)
        if x1_grid is None:
            raise ValueError("If y_line_hist is provided, x1_grid must be provided.")
        x1_grid = np.asarray(x1_grid, dtype=float).ravel()

        if y_line_hist.ndim != 2:
            raise ValueError("y_line_hist must have shape (steps, grid_points).")
        if y_line_hist.shape[1] != x1_grid.size:
            raise ValueError("y_line_hist second dim must match x1_grid size.")
        steps_n = int(y_line_hist.shape[0])

        def y_line(t: int):
            return y_line_hist[t]

        # If theta history is also provided, show numeric equation; else show generic
        w_disp = None
        b_disp = None
        if w_hist is not None and b_hist is not None:
            w_arr = np.asarray(w_hist, dtype=float)
            b_arr = np.asarray(b_hist, dtype=float).ravel()

            # accept shapes: (T,), (T,1)
            if w_arr.ndim == 2 and w_arr.shape[1] == 1:
                w_arr = w_arr[:, 0]
            if w_arr.ndim == 1 and w_arr.size == steps_n and b_arr.size == steps_n:
                w_disp = w_arr
                b_disp = b_arr

        def theta_formula_text():
            return r"$\hat{y}=\theta_0+\theta_1 x_1$"

        def eq_text(t: int):
            if w_disp is None:
                return r"$\hat{y} = f(x_1)$"
            return rf"$\hat{{y}} = ({w_disp[t]:.{dec}f})x_1 + ({b_disp[t]:.{dec}f})$"

        x_min, x_max = float(x1_grid.min()), float(x1_grid.max())

    else:
        # legacy path
        if w_hist is None or b_hist is None:
            raise ValueError("Legacy mode requires w_hist and b_hist. Prefer providing y_line_hist + x1_grid.")

        w_hist = np.asarray(w_hist, dtype=float)
        b_hist = np.asarray(b_hist, dtype=float).ravel()
        steps_n = int(b_hist.size)

        # allow w_hist shape flexibility
        if w_hist.ndim == 1:
            w_hist = w_hist.reshape(-1, 1)
        if w_hist.shape[0] != steps_n:
            raise ValueError("w_hist and b_hist must have the same number of steps.")
        if w_hist.shape[1] != 1:
            raise ValueError(f"Simple LR expects 1 weight, got d={w_hist.shape[1]}.")

        x_min, x_max = float(x1.min()), float(x1.max())
        x1_grid = np.linspace(x_min, x_max, 250)

        def y_line(t: int):
            w1 = float(w_hist[t, 0])
            b = float(b_hist[t])
            return w1 * x1_grid + b

        def theta_formula_text():
            return r"$\hat{y}=\theta_0+\theta_1 x_1$"

        def eq_text(t: int):
            w1 = float(w_hist[t, 0])
            b = float(b_hist[t])
            return rf"$\hat{{y}} = ({w1:.{dec}f})x_1 + ({b:.{dec}f})$"

    if steps_n < 1:
        raise ValueError("Need at least 1 step to animate.")

    # validate loss
    if show_loss:
        if loss_hist is None:
            raise ValueError("show_loss=True requires loss_hist.")
        loss_hist = np.asarray(loss_hist, dtype=float).ravel()
        if loss_hist.size != steps_n:
            raise ValueError("loss_hist must have the same length as steps.")

    step_axis = np.arange(steps_n)

    # -------------------------
    # Annotations (paper coords)
    # -------------------------
    # NOTE: for subplots we use xref/yref="paper" too; it's fine because it's global paper.
    if show_loss:
        theta_y = 1.18
        eq_y = 1.10
        margin_t = 160
    else:
        theta_y = 1.15
        eq_y = 1.05
        margin_t = 150

    def theta_formula_annotation():
        return dict(
            x=0.5,
            y=theta_y,
            xref="paper",
            yref="paper",
            text=theta_formula_text(),
            showarrow=False,
            xanchor="center",
            yanchor="bottom",
            font=dict(color="white", size=16),
        )

    def eq_annotation(t):
        return dict(
            x=0.5,
            y=eq_y,
            xref="paper",
            yref="paper",
            text=eq_text(t),
            showarrow=False,
            xanchor="center",
            yanchor="bottom",
            font=dict(color="white", size=16),
        )

    # -------------------------
    # Stable ranges
    # -------------------------
    # (use step 0 and last to stabilize y-range)
    y_all = np.concatenate(
        [
            y,
            np.asarray(y_line(0)).ravel(),
            np.asarray(y_line(steps_n - 1)).ravel(),
        ]
    )
    y_min, y_max = float(y_all.min()), float(y_all.max())
    y_pad = 0.08 * (y_max - y_min + 1e-9)

    def _pad(lo, hi, frac=0.10):
        span = (hi - lo) + 1e-9
        return [lo - frac * span, hi + frac * span]

    x_range = _pad(x_min, x_max)

    if show_loss:
        lmin, lmax = float(loss_hist.min()), float(loss_hist.max())
        lpad = 0.10 * (lmax - lmin + 1e-9)

    # =====================================================================
    # CASE A) show_loss=True
    # =====================================================================
    if show_loss:
        fig = make_subplots(
            rows=1,
            cols=2,
            column_widths=[0.62, 0.38],
            horizontal_spacing=0.08,
            specs=[[{"type": "xy"}, {"type": "xy"}]],
        )

        # Data
        fig.add_trace(
            go.Scatter(
                x=x1,
                y=y,
                mode="markers",
                name="Data",
                marker=dict(size=7, opacity=0.85),
                legendgroup="fit",
                showlegend=True,
            ),
            row=1,
            col=1,
        )

        # Model line
        fig.add_trace(
            go.Scatter(
                x=x1_grid,
                y=y_line(0),
                mode="lines",
                name="Model",
                line=dict(width=4),
                legendgroup="fit",
                showlegend=True,
                uid="MODEL_LINE",
            ),
            row=1,
            col=1,
        )

        # Loss line (start as a single point)
        fig.add_trace(
            go.Scatter(
                x=[0],
                y=[loss_hist[0]],
                mode="lines",
                name="Loss",
                line=dict(width=3),  # don't hardcode colors
                legendgroup="loss",
                showlegend=True,
                uid="LOSS_LINE",
            ),
            row=1,
            col=2,
        )

        # Frames
        frames = []
        for t in range(steps_n):
            frames.append(
                go.Frame(
                    name=str(t),
                    data=[
                        go.Scatter(x=x1_grid, y=y_line(t), mode="lines", line=dict(width=4), uid="MODEL_LINE"),
                        go.Scatter(
                            x=step_axis[: t + 1],
                            y=loss_hist[: t + 1],
                            mode="lines",
                            line=dict(width=3),
                            uid="LOSS_LINE",
                        ),
                    ],
                    traces=[1, 2],  # update model + loss
                    layout=go.Layout(annotations=[theta_formula_annotation(), eq_annotation(t)]),
                )
            )
        fig.frames = frames

        fig.update_layout(
            template="plotly_dark",
            height=720,
            font=dict(family="Helvetica", color="white"),
            title=dict(
                text=title,
                y=0.96,
                x=0.5,
                xanchor="center",
                font=dict(color="white", size=24),
            ),
            annotations=[theta_formula_annotation(), eq_annotation(0)],
            margin=dict(t=margin_t, r=30, l=60, b=70),
            legend=dict(
                orientation="v",
                x=0.49,
                y=0.02,
                xanchor="right",
                yanchor="bottom",
                bgcolor="rgba(220,220,220,0.85)",
                bordercolor="rgba(0,0,0,0.6)",
                borderwidth=1,
                font=dict(size=12, color="black"),
            ),
            legend2=dict(
                orientation="v",
                x=0.985,
                y=0.02,
                xanchor="right",
                yanchor="bottom",
                bgcolor="rgba(220,220,220,0.85)",
                bordercolor="rgba(0,0,0,0.6)",
                borderwidth=1,
                font=dict(size=12, color="black"),
            ),
            sliders=[
                dict(
                    active=0,
                    currentvalue=dict(prefix="Step: "),
                    pad=dict(t=55),
                    steps=[
                        dict(
                            method="animate",
                            args=[
                                [str(t)],
                                {
                                    "mode": "immediate",
                                    "frame": {"duration": 0, "redraw": True},
                                    "transition": {"duration": 0},
                                },
                            ],
                            label=str(t),
                        )
                        for t in range(steps_n)
                    ],
                )
            ],
            updatemenus=[
                dict(
                    type="buttons",
                    direction="left",
                    x=0.10,
                    y=1.14,
                    bgcolor="white",
                    bordercolor="black",
                    borderwidth=1,
                    font=dict(color="black", size=14),
                    buttons=[
                        dict(
                            label="Play",
                            method="animate",
                            args=[None, {"frame": {"duration": 80, "redraw": True}, "transition": {"duration": 0}}],
                        ),
                        dict(
                            label="Pause",
                            method="animate",
                            args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
                        ),
                    ],
                )
            ],
        )

        # Put loss on legend2
        fig.data[2].update(legend="legend2")

        fig.update_xaxes(title="x₁", range=x_range, row=1, col=1)
        fig.update_yaxes(title="y", range=[y_min - y_pad, y_max + y_pad], row=1, col=1)

        fig.update_xaxes(title="Step", range=[0, steps_n - 1], row=1, col=2)
        fig.update_yaxes(title="Loss", range=[lmin - lpad, lmax + lpad], row=1, col=2)

        return fig

    # =====================================================================
    # CASE B) show_loss=False
    # =====================================================================
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x1,
            y=y,
            mode="markers",
            name="Data",
            marker=dict(size=7, opacity=0.85),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x1_grid,
            y=y_line(0),
            mode="lines",
            name="Model",
            line=dict(width=4),
            uid="MODEL_LINE",
        )
    )

    frames = []
    for t in range(steps_n):
        frames.append(
            go.Frame(
                name=str(t),
                data=[go.Scatter(x=x1_grid, y=y_line(t), mode="lines", line=dict(width=4), uid="MODEL_LINE")],
                traces=[1],
                layout=go.Layout(annotations=[theta_formula_annotation(), eq_annotation(t)]),
            )
        )
    fig.frames = frames

    fig.update_layout(
        template="plotly_dark",
        height=720,
        font=dict(family="Helvetica", color="white"),
        title=dict(
            text=title,
            y=0.96,
            x=0.5,
            xanchor="center",
            font=dict(color="white", size=24),
        ),
        annotations=[theta_formula_annotation(), eq_annotation(0)],
        margin=dict(t=margin_t, r=60, l=70, b=80),
        legend=dict(
            x=0.985,
            y=0.02,
            xanchor="right",
            yanchor="bottom",
            bgcolor="rgba(220,220,220,0.85)",
            bordercolor="rgba(0,0,0,0.6)",
            borderwidth=1,
            font=dict(color="black", size=12),
        ),
        xaxis=dict(title="x₁", range=x_range),
        yaxis=dict(title="y", range=[y_min - y_pad, y_max + y_pad]),
        sliders=[
            dict(
                active=0,
                currentvalue=dict(prefix="Step: "),
                pad=dict(t=55),
                steps=[
                    dict(
                        method="animate",
                        args=[
                            [str(t)],
                            {
                                "mode": "immediate",
                                "frame": {"duration": 0, "redraw": True},
                                "transition": {"duration": 0},
                            },
                        ],
                        label=str(t),
                    )
                    for t in range(steps_n)
                ],
            )
        ],
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                x=0.10,
                y=1.14,
                bgcolor="white",
                bordercolor="black",
                borderwidth=1,
                font=dict(color="black", size=14),
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[None, {"frame": {"duration": 80, "redraw": True}, "transition": {"duration": 0}}],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
                    ),
                ],
            )
        ],
    )

    return fig


def build_lr_figure(
    X,
    y,
    w_hist=None,
    b_hist=None,
    *,
    history=None,
    y_line_hist=None,
    x1_grid=None,  # d==1
    z_plane_hist=None,
    X1g=None,
    X2g=None,  # d==2
    loss_hist=None,
    show_loss=False,
    history_kind="iterative",
    title=None,
    strict_loss=False,
    dec=4,  # passthrough a figuras (opcional, pero útil)
):
    """
    Route to the appropriate visualization figure based on feature dimensions.

    Depending on the number of features `d` in the dataset `X`, this function delegates
    the plot creation to the respective builder for 1D, 2D, or multivariable data.

    Args:
        X (np.ndarray): The feature matrix of shape (n_samples, d).
        y (np.ndarray): The target vector of shape (n_samples,).
        w_hist (np.ndarray, optional): History of weights (theta). Defaults to None.
        b_hist (np.ndarray, optional): History of biases (intercepts). Defaults to None.
        history (dict, optional): Complete history dictionary returned by `fit_history()`. Defaults to None.
        y_line_hist (np.ndarray, optional): History of prediction lines (for 1D). Defaults to None.
        x1_grid (np.ndarray, optional): X-axis grid for 1D predictions. Defaults to None.
        z_plane_hist (np.ndarray, optional): History of prediction planes (for 2D). Defaults to None.
        X1g (np.ndarray, optional): Grid for first feature in 2D. Defaults to None.
        X2g (np.ndarray, optional): Grid for second feature in 2D. Defaults to None.
        loss_hist (np.ndarray, optional): History of loss values. Defaults to None.
        show_loss (bool, optional): Whether to display the loss chart. Defaults to False.
        history_kind (str, optional): The kind of history collected ("iterative" or "auto"). Defaults to "iterative".
        title (str, optional): The main title of the figure. Defaults to None.
        strict_loss (bool, optional): If True, strictly enforce loss display rules. Defaults to False.
        dec (int, optional): Number of decimal places to show for parameters. Defaults to 4.

    Returns:
        plotly.graph_objects.Figure: The fully constructed Plotly figure.
    """
    X = np.asarray(X)
    y = np.asarray(y).ravel()
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    d = int(X.shape[1])

    # ---- history dict (sin OR con arrays) ----
    if history is not None:
        if not isinstance(history, dict):
            raise ValueError("history must be a dict returned by fit_history().")

        history_kind = history.get("history_kind", history_kind)

        # NO uses: a or b or c  (si a es np.array revienta)
        loss_hist = _first_not_none(
            history.get("loss_hist", None),
            history.get("losses", None),
            history.get("loss", None),
            loss_hist,
        )

        grid = history.get("grid", {}) or {}

        # Prefer theta "para mostrar" (respeta display_space de fit_history)
        w_hist = _first_not_none(history.get("w_hist", None), w_hist)
        b_hist = _first_not_none(history.get("b_hist", None), b_hist)

        if d == 1:
            y_line_hist = _first_not_none(history.get("y_line_hist", None), y_line_hist)
            x1_grid = _first_not_none(grid.get("x1_grid", None), x1_grid)

        elif d == 2:
            z_plane_hist = _first_not_none(history.get("z_plane_hist", None), z_plane_hist)
            X1g = _first_not_none(grid.get("X1g", None), X1g)
            X2g = _first_not_none(grid.get("X2g", None), X2g)

    # ---- routing ----
    if d == 1:
        x1 = X[:, 0]
        if title is None:
            title = "Linear Regression (Simple, 1 variable)"
        return build_simple_lr_figure(
            x1,
            y,
            w_hist=w_hist,
            b_hist=b_hist,
            y_line_hist=y_line_hist,
            x1_grid=x1_grid,
            loss_hist=loss_hist,
            show_loss=show_loss,
            history_kind=history_kind,
            title=title,
            strict_loss=strict_loss,
            dec=dec,
        )

    if d == 2:
        x1 = X[:, 0]
        x2 = X[:, 1]
        if title is None:
            title = "Linear Regression (2 variables)"
        return build_plane_lr_figure(
            x1,
            x2,
            y,
            w_hist=w_hist,
            b_hist=b_hist,
            z_plane_hist=z_plane_hist,
            X1g=X1g,
            X2g=X2g,
            loss_hist=loss_hist,
            show_loss=show_loss,
            history_kind=history_kind,
            title=title,
            strict_loss=strict_loss,
            dec=dec,
        )

    if d > 2:
        # Para d>2 esta vista ES theta-based (no hay "pred-grid" equivalente aquí)
        if w_hist is None or b_hist is None:
            raise ValueError("For d>2, this visualization expects w_hist and b_hist (parameter-display-based).")
        if title is None:
            title = f"Multivariable Linear Regression Model ({d} variables)"
        return build_multivar_lr_figure(
            X,
            y,
            w_hist,
            b_hist,
            loss_hist=loss_hist,
            show_loss=show_loss,
            history_kind=history_kind,
            title=title,
            strict_loss=strict_loss,
            dec=dec,
        )

    raise ValueError(f"Unexpected d={d}.")


# ============================================================
# API PÚBLICA (lo único que usa el usuario)
# ============================================================
def visualize_lr(
    trained_estimator,
    X,
    y,
    *,
    steps=60,
    mode="auto",
    show_loss=True,
    title=None,
    smooth="ema",
    smooth_beta=0.85,
    strict_loss=False,
    baseline="mean",
    display_space="original",  # <-- NUEVO
    dec=4,  # passthrough (opcional)
):
    """
    Generate an animated visualization for a linear regression model.

    This function is the primary public API of the library. It extracts the training
    history from the provided scikit-learn estimator and creates an interactive
    Plotly animation that demonstrates the evolution of the model's parameters and
    predictions across training steps.

    Args:
        trained_estimator: A fitted scikit-learn estimator or Pipeline.
        X (np.ndarray): The feature matrix used for training.
        y (np.ndarray): The target vector.
        steps (int, optional): The desired number of animation frames. Defaults to 60.
        mode (str, optional): Method to extract history ("auto", "iterative", "final_interp"). Defaults to "auto".
        show_loss (bool, optional): Whether to display the loss curve alongside the main plot. Defaults to True.
        title (str, optional): The title of the plot. Defaults to None.
        smooth (str, optional): Smoothing method for the loss curve (e.g., "ema" or None). Defaults to "ema".
        smooth_beta (float, optional): Beta parameter for EMA smoothing. Defaults to 0.85.
        strict_loss (bool, optional): If True, throw errors if loss cannot be animated cleanly. Defaults to False.
        baseline (str, optional): Initial reference line for the loss curve ("mean" or "zeros"). Defaults to "mean".
        display_space (str, optional): The space in which to display the parameters ("original" or "scaled"). Defaults to "original".
        dec (int, optional): The number of decimal places to format the parameters. Defaults to 4.

    Returns:
        plotly.graph_objects.Figure: The animated Plotly figure object.
    """
    hist = fit_history(
        trained_estimator,
        X,
        y,
        steps=steps,
        mode=mode,
        smooth=smooth,
        smooth_beta=smooth_beta,
        baseline=baseline,
        display_space=display_space,  # <-- NUEVO
    )

    return build_lr_figure(
        X,
        y,
        history=hist,
        show_loss=show_loss,
        title=title,
        strict_loss=strict_loss,
        dec=dec,
    )


def build_multivar_lr_figure(
    X,
    y,
    w_hist,
    b_hist,
    *,
    loss_hist=None,
    show_loss=True,
    history_kind="iterative",
    title=None,
    strict_loss=False,
    terms_per_line=6,
    dec=4,
    threshold_dense=100,  # <=100 usa expansión completa; >100 usa vista matricial
):
    """
    Multivariable visualization for d > 2 (parameter display).

    Important:
    - This visualization is inherently tied to showing weights (theta).
    - If the user's model uses arbitrary transforms/pipelines, theta in original space may not be meaningful.
    """
    # --- enforce inside the library ---
    if show_loss and history_kind != "iterative":
        if strict_loss:
            raise ValueError("show_loss=True is only allowed for iterative histories.")
        show_loss = False
        loss_hist = None

    X = np.asarray(X)
    y = np.asarray(y).ravel()
    w_hist = np.asarray(w_hist, dtype=float)
    b_hist = np.asarray(b_hist, dtype=float).ravel()

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    # Validate shapes early (robust to 1D w_hist)
    if w_hist.ndim == 1:
        # Allow flatten only if it can be inferred from b_hist
        steps_n = int(b_hist.size)
        if steps_n < 1:
            raise ValueError("Need at least 1 step to animate.")
        if w_hist.size % steps_n != 0:
            raise ValueError("w_hist is 1D but cannot be reshaped to (steps, d) using b_hist length.")
        d = int(w_hist.size // steps_n)
        w_hist = w_hist.reshape(steps_n, d)
    else:
        steps_n = int(b_hist.size)
        if steps_n < 1:
            raise ValueError("Need at least 1 step to animate.")
        if w_hist.ndim != 2:
            raise ValueError("w_hist must have shape (steps, d).")
        if w_hist.shape[0] != steps_n:
            raise ValueError("w_hist must have shape (steps, d) and match b_hist length.")
        d = int(w_hist.shape[1])

    d_X = int(X.shape[1])
    if d_X != d:
        raise ValueError(
            f"X has d={d_X} features but w_hist has d={d}. For d>2 visualization we require theta compatible with X."
        )

    if d <= 2:
        raise ValueError("This figure is intended for d > 2. Use 1D/2D figures for d<=2.")

    if title is None:
        title = f"Multivariable Linear Regression Model ({d} variables)"

    if show_loss:
        if loss_hist is None:
            raise ValueError("show_loss=True requires loss_hist.")
        loss_hist = np.asarray(loss_hist, dtype=float).ravel()
        if loss_hist.size != steps_n:
            raise ValueError("loss_hist must have the same length as b_hist.")

    step_axis = np.arange(steps_n)

    # Stable ranges (loss)
    if show_loss:
        lmin, lmax = float(loss_hist.min()), float(loss_hist.max())
        lpad = 0.08 * ((lmax - lmin) + 1e-9)

    # ------------------------------------------------------------
    # Helper: detect "big" coefficients (more than 5 integer digits)
    # ------------------------------------------------------------
    def _needs_single_col(values, max_digits=5):
        vals = np.asarray(values, dtype=float).ravel()
        for v in vals:
            if not np.isfinite(v):
                return True
            # count digits of integer part
            int_digits = len(str(int(abs(float(v)))))
            if int_digits > max_digits:
                return True
        return False

    def _theta_is_big_for_t(t: int):
        return _needs_single_col(w_hist[t], max_digits=5)

    # ------------------------------------------------------------
    # Decide mode for <= threshold_dense:
    # - default: MODE A (full expansion)
    # - if any coef has >5 integer digits: force "matrix view" (stable layout)
    # ------------------------------------------------------------
    force_matrix_for_dense = False
    if d <= threshold_dense:
        for t in range(steps_n):
            if _theta_is_big_for_t(t):
                force_matrix_for_dense = True
                break

    # =====================================================================
    # MODE A) 3..threshold_dense: full expansion (OR forced matrix if big)
    # =====================================================================
    if d <= threshold_dense and not force_matrix_for_dense:

        def model_header_latex():
            return rf"$$\hat{{y}} = \sum_{{j=1}}^{{{d}}} \theta_j x_j + \theta_0$$"

        def full_scalar_model_multiline_latex(t: int):
            w = w_hist[t]
            b = float(b_hist[t])

            terms = [rf"({w[i]:.{dec}f})x_{{{i + 1}}}" for i in range(d)]
            chunks = [terms[i : i + terms_per_line] for i in range(0, len(terms), terms_per_line)]

            lines = []
            lines.append(r"\hat{y} = " + " + ".join(chunks[0]))
            for ch in chunks[1:]:
                lines.append(r"\quad " + " + ".join(ch))

            lines[-1] = lines[-1] + rf" + ({b:.{dec}f})"
            body = r" \\ ".join(lines)
            return r"$$\begin{aligned}" + body + r"\end{aligned}$$"

        def make_annotations(t: int):
            ann = [
                dict(
                    x=0.68,
                    y=0.93,
                    xref="paper",
                    yref="paper",
                    text=model_header_latex(),
                    showarrow=False,
                    xanchor="center",
                    yanchor="top",
                    font=dict(size=22, color="white"),
                ),
                dict(
                    x=0.68,
                    y=0.78,
                    xref="paper",
                    yref="paper",
                    text=full_scalar_model_multiline_latex(t),
                    showarrow=False,
                    xanchor="center",
                    yanchor="top",
                    font=dict(size=17, color="white"),
                ),
            ]

            if show_loss:
                ann.append(
                    dict(
                        x=0.33,
                        y=0.94,
                        xref="paper",
                        yref="paper",
                        text=f"<b>Loss</b><br>{loss_hist[t]:.6f}",
                        showarrow=False,
                        xanchor="left",
                        yanchor="top",
                        font=dict(size=16, color="black"),
                        bgcolor="white",
                        bordercolor="black",
                        borderwidth=1,
                        borderpad=8,
                    )
                )
            return ann

        fig = make_subplots(
            rows=1,
            cols=2,
            column_widths=[0.42, 0.58],
            horizontal_spacing=0.06,
            specs=[[{"type": "xy"}, {"type": "xy"}]],
        )

        fig.add_trace(
            go.Scatter(
                x=[],
                y=[],
                mode="lines",
                name="Loss",
                line=dict(width=3),  # don't hardcode color
                uid="LOSS_LINE",
            ),
            row=1,
            col=1,
        )

        frames = []
        for t in range(steps_n):
            if show_loss:
                loss_trace = go.Scatter(
                    x=step_axis[: t + 1],
                    y=loss_hist[: t + 1],
                    mode="lines",
                    line=dict(width=3),
                    uid="LOSS_LINE",
                )
            else:
                loss_trace = go.Scatter(x=[], y=[], uid="LOSS_LINE")

            frames.append(
                go.Frame(
                    name=str(t),
                    data=[loss_trace],
                    traces=[0],
                    layout=go.Layout(annotations=make_annotations(t)),
                )
            )
        fig.frames = frames

        fig.update_layout(
            template="plotly_dark",
            height=760,
            font=dict(family="Helvetica"),
            title=dict(
                text=title,
                x=0.5,
                xanchor="center",
                font=dict(color="white", size=24),
            ),
            margin=dict(l=70, r=40, t=110, b=95),
            showlegend=False,
            sliders=[
                dict(
                    active=0,
                    currentvalue=dict(prefix="Step: "),
                    pad=dict(t=45),
                    steps=[
                        dict(
                            method="animate",
                            args=[
                                [str(t)],
                                {
                                    "mode": "immediate",
                                    "frame": {"duration": 0, "redraw": True},
                                    "transition": {"duration": 0},
                                },
                            ],
                            label=str(t),
                        )
                        for t in range(steps_n)
                    ],
                )
            ],
            updatemenus=[
                dict(
                    type="buttons",
                    direction="left",
                    x=0.07,
                    y=1.1,
                    xanchor="left",
                    yanchor="top",
                    bgcolor="white",
                    bordercolor="black",
                    borderwidth=1,
                    font=dict(color="black", size=14),
                    buttons=[
                        dict(
                            label="Play",
                            method="animate",
                            args=[None, {"frame": {"duration": 80, "redraw": True}, "transition": {"duration": 0}}],
                        ),
                        dict(
                            label="Pause",
                            method="animate",
                            args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
                        ),
                    ],
                )
            ],
            annotations=make_annotations(0),
        )

        fig.update_xaxes(title="Step", range=[0, steps_n - 1], row=1, col=1)
        if show_loss:
            fig.update_yaxes(title="Loss", range=[lmin - lpad, lmax + lpad], row=1, col=1)
        else:
            fig.update_yaxes(title="Loss", row=1, col=1)

        fig.update_xaxes(visible=False, row=1, col=2, range=[0, 1])
        fig.update_yaxes(visible=False, row=1, col=2, range=[0, 1])

        return fig

    # =====================================================================
    # MODE B) matrix view (d > threshold_dense) + forced-matrix for dense if big
    # =====================================================================

    rows = 15
    x_cols = 5
    capacity_x = rows * x_cols

    # For theta columns:
    # - if "big" coef anywhere => force theta to 1 col (d×1)
    # - else: keep theta in 5 cols
    force_theta_one_col = False
    for t in range(steps_n):
        if _theta_is_big_for_t(t):
            force_theta_one_col = True
            break

    def theta_cols_for_t(_t: int):
        return 1 if force_theta_one_col else 5

    def model_formula_latex():
        return r"$$\hat{y} = \theta_0 + \operatorname{vec}(\boldsymbol{\theta})^\top \operatorname{vec}(\mathbf{x})$$"

    def bias_latex(t: int):
        return rf"$$\theta_0 = {float(b_hist[t]):.{dec}f}$$"

    def x_dim_latex():
        return rf"$$\mathbf{{x}} \in \mathbb{{R}}^{{{d}\times {x_cols}}}$$"

    def theta_dim_latex(t: int):
        th_cols = theta_cols_for_t(t)
        return rf"$$\boldsymbol{{\theta}} \in \mathbb{{R}}^{{{d}\times {th_cols}}}$$"

    # -----------------------------
    # X matrix
    # -----------------------------
    def x_vector_latex():
        def cell(j):
            return rf"x_{{{j}}}"

        def vdots_row():
            return " & ".join([r"\vdots"] * x_cols)

        lines = []
        if d <= capacity_x:
            items = [cell(j) for j in range(1, d + 1)] + [r"\;"] * (capacity_x - d)
            M = np.array(items, dtype=object).reshape(rows, x_cols)
            for r in range(rows):
                lines.append(" & ".join(M[r, c] for c in range(x_cols)))
        else:
            head_rows = rows // 2
            tail_rows = rows - head_rows - 1

            head_js = list(range(1, head_rows * x_cols + 1))
            H = np.array([cell(j) for j in head_js], dtype=object).reshape(head_rows, x_cols)
            for r in range(head_rows):
                lines.append(" & ".join(H[r, c] for c in range(x_cols)))

            lines.append(vdots_row())

            tail_count = tail_rows * x_cols
            tail_js = list(range(d - tail_count + 1, d + 1))
            T = np.array([cell(j) for j in tail_js], dtype=object).reshape(tail_rows, x_cols)
            for r in range(tail_rows):
                lines.append(" & ".join(T[r, c] for c in range(x_cols)))

        body = r" \\ ".join(lines)
        return rf"$$\mathbf{{x}} = \begin{{bmatrix}} {body} \end{{bmatrix}}$$"

    # -----------------------------
    # Theta matrix
    # -----------------------------
    def w_matrix_latex(t: int):
        w = np.asarray(w_hist[t], dtype=float).ravel()
        d_local = w.size

        th_cols = theta_cols_for_t(t)

        def fmt(x):
            return rf"{x:+.{dec}f}"

        # ---- Case: theta ONE COLUMN (d×1) ----
        if th_cols == 1:
            if d_local <= rows:
                lines = [fmt(w[i]) for i in range(d_local)] + [r"\;"] * (rows - d_local)
            else:
                head_rows = rows // 2
                tail_rows = rows - head_rows - 1

                head_vals = w[:head_rows]
                tail_vals = w[-tail_rows:]

                lines = [fmt(v) for v in head_vals]
                lines.append(r"\vdots")
                lines += [fmt(v) for v in tail_vals]

            body = r" \\ ".join(lines)
            return rf"$$\boldsymbol{{\theta}} = \begin{{bmatrix}} {body} \end{{bmatrix}}$$"

        # ---- Case: theta 5 columns ----
        th_capacity = rows * th_cols

        def vdots_row():
            return " & ".join([r"\vdots"] * th_cols)

        lines = []
        if d_local <= th_capacity:
            padded = np.full(th_capacity, np.nan, dtype=float)
            padded[:d_local] = w
            W = padded.reshape(rows, th_cols)

            for r in range(rows):
                row_items = []
                for c in range(th_cols):
                    if np.isnan(W[r, c]):
                        row_items.append(r"\;")
                    else:
                        row_items.append(fmt(W[r, c]))
                lines.append(" & ".join(row_items))
        else:
            head_rows = rows // 2
            tail_rows = rows - head_rows - 1

            head_vals = w[: head_rows * th_cols]
            tail_vals = w[-(tail_rows * th_cols) :]

            H = head_vals.reshape(head_rows, th_cols)
            for r in range(head_rows):
                lines.append(" & ".join(fmt(H[r, c]) for c in range(th_cols)))

            lines.append(vdots_row())

            T = tail_vals.reshape(tail_rows, th_cols)
            for r in range(tail_rows):
                lines.append(" & ".join(fmt(T[r, c]) for c in range(th_cols)))

        body = r" \\ ".join(lines)
        return rf"$$\boldsymbol{{\theta}} = \begin{{bmatrix}} {body} \end{{bmatrix}}$$"

    # -----------------------------
    # Compact equation below matrices
    # -----------------------------
    def scalar_model_compact_latex(t: int):
        w = np.asarray(w_hist[t], dtype=float).ravel()
        b = float(b_hist[t])
        last = d

        th_cols = theta_cols_for_t(t)

        if th_cols == 1:
            return (
                r"$$\hat{y} = "
                + rf"({w[0]:.{dec}f})x_1 "
                + rf"+ \cdots + ({w[last - 1]:.{dec}f})x_{{{last}}} "
                + rf"+ ({b:.{dec}f}) $$"
            )

        return (
            r"$$\hat{y} = "
            + rf"({w[0]:.{dec}f})x_1 "
            + rf"+ ({w[1]:.{dec}f})x_2 "
            + rf"+ ({w[2]:.{dec}f})x_3 "
            + rf"+ ({w[3]:.{dec}f})x_4 "
            + rf"+ \cdots + ({w[last - 1]:.{dec}f})x_{{{last}}} "
            + rf"+ ({b:.{dec}f}) $$"
        )

    def make_annotations(t: int):
        ann = [
            dict(
                x=0.68,
                y=0.995,
                xref="paper",
                yref="paper",
                text=model_formula_latex(),
                showarrow=False,
                xanchor="center",
                yanchor="top",
                font=dict(size=22, color="white"),
            ),
            dict(
                x=0.68,
                y=0.938,
                xref="paper",
                yref="paper",
                text=bias_latex(t),
                showarrow=False,
                xanchor="center",
                yanchor="top",
                font=dict(size=18, color="white"),
            ),
            dict(
                x=0.55,
                y=0.83,
                xref="paper",
                yref="paper",
                text=x_dim_latex(),
                showarrow=False,
                xanchor="center",
                yanchor="bottom",
                font=dict(size=14, color="white"),
            ),
            dict(
                x=0.83,
                y=0.83,
                xref="paper",
                yref="paper",
                text=theta_dim_latex(t),
                showarrow=False,
                xanchor="center",
                yanchor="bottom",
                font=dict(size=14, color="white"),
            ),
            dict(
                x=0.52,
                y=0.48,
                xref="paper",
                yref="paper",
                text=x_vector_latex(),
                showarrow=False,
                xanchor="center",
                yanchor="middle",
                font=dict(size=15, color="white"),
            ),
            dict(
                x=0.80,
                y=0.48,
                xref="paper",
                yref="paper",
                text=w_matrix_latex(t),
                showarrow=False,
                xanchor="center",
                yanchor="middle",
                font=dict(size=15, color="white"),
            ),
            dict(
                x=0.71,
                y=0.03,
                xref="paper",
                yref="paper",
                text=scalar_model_compact_latex(t),
                showarrow=False,
                xanchor="center",
                yanchor="middle",
                font=dict(size=16, color="white"),
            ),
        ]

        if show_loss:
            th_cols = theta_cols_for_t(t)
            y_loss = 0.98 if th_cols == 1 else 0.86

            ann.append(
                dict(
                    x=0.25,
                    y=y_loss,
                    xref="paper",
                    yref="paper",
                    text=f"<b>Loss</b><br>{loss_hist[t]:.6f}",
                    showarrow=False,
                    xanchor="left",
                    yanchor="top",
                    font=dict(size=16, color="black"),
                    bgcolor="white",
                    bordercolor="black",
                    borderwidth=1,
                    borderpad=8,
                )
            )

        return ann

    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.42, 0.58],
        horizontal_spacing=0.06,
        specs=[[{"type": "xy"}, {"type": "xy"}]],
    )

    fig.add_trace(
        go.Scatter(
            x=[],
            y=[],
            mode="lines",
            name="Loss",
            line=dict(width=3),  # don't hardcode color
            uid="LOSS_LINE",
        ),
        row=1,
        col=1,
    )

    frames = []
    for t in range(steps_n):
        if show_loss:
            loss_trace = go.Scatter(
                x=step_axis[: t + 1],
                y=loss_hist[: t + 1],
                mode="lines",
                line=dict(width=3),
                uid="LOSS_LINE",
            )
        else:
            loss_trace = go.Scatter(x=[], y=[], uid="LOSS_LINE")

        frames.append(
            go.Frame(
                name=str(t),
                data=[loss_trace],
                traces=[0],
                layout=go.Layout(annotations=make_annotations(t)),
            )
        )
    fig.frames = frames

    fig.update_layout(
        template="plotly_dark",
        font=dict(family="Helvetica"),
        height=760,
        title=dict(
            text=title,
            x=0.5,
            xanchor="center",
            font=dict(color="white", size=24),
        ),
        margin=dict(l=70, r=40, t=110, b=95),
        showlegend=False,
        sliders=[
            dict(
                active=0,
                currentvalue=dict(prefix="Step: "),
                pad=dict(t=45),
                steps=[
                    dict(
                        method="animate",
                        args=[
                            [str(t)],
                            {
                                "mode": "immediate",
                                "frame": {"duration": 0, "redraw": True},
                                "transition": {"duration": 0},
                            },
                        ],
                        label=str(t),
                    )
                    for t in range(steps_n)
                ],
            )
        ],
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                x=0.07,
                y=1.1,
                xanchor="left",
                yanchor="top",
                bgcolor="white",
                bordercolor="black",
                borderwidth=1,
                font=dict(color="black", size=14),
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[None, {"frame": {"duration": 80, "redraw": True}, "transition": {"duration": 0}}],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
                    ),
                ],
            )
        ],
        annotations=make_annotations(0),
    )

    fig.update_xaxes(title="Step", range=[0, steps_n - 1], row=1, col=1)
    if show_loss:
        fig.update_yaxes(title="Loss", range=[lmin - lpad, lmax + lpad], row=1, col=1)
    else:
        fig.update_yaxes(title="Loss", row=1, col=1)

    fig.update_xaxes(visible=False, row=1, col=2, range=[0, 1])
    fig.update_yaxes(visible=False, row=1, col=2, range=[0, 1])

    return fig
