First rigorous logistic-regression lesson
============================================================

Binary logistic regression connects a linear score to the positive fitted-class
probability:

.. math::

   z=\theta_0+\mathbf{x}^{\mathsf T}\boldsymbol\theta,
   \qquad p_1=\sigma(z)=\frac{1}{1+e^{-z}}.

The class decision uses a documented threshold. Class indices follow
``estimator.classes_`` exactly. Semantic labels are opt-in presentation through
``show_class_labels=True``; they never change the probability mathematics.

.. code-block:: python

   fig = visualize_logistic(
       model,
       X,
       y,
       threshold=0.65,
       show_loss=True,
       detail="complete",
       show_class_labels=False,
   )

For multiclass estimators, Mlektic resolves supported multinomial Softmax or
normalized one-vs-rest sigmoid semantics and preserves fitted class order.
``class_focus`` reduces visible surface clutter without discarding the complete
probability vector from metadata.

What is exact?
==============

Fitted scores, ``predict_proba`` values, class order, and the final state are
model-verified. Scikit-learn does not expose its original private optimization
trajectory, so intermediate states retain explicit replay or interpolation
labels.

Inspect ``LEARN-LOGISTIC-BINARY`` and ``LEARN-LOGISTIC-MULTI`` in
``notebooks/learn/learn_02_logistic_regression.ipynb``.
