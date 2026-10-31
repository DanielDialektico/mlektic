========================
Mathematical conventions
========================
“rejected.” Inspect ``LR-POLYNOMIAL`` and ``LOG-MULTI-FOCUS`` in the QA suite.
Mlektic uses :math:`\theta_0` for an intercept, bold lower-case symbols for
vectors, upper-case symbols for matrices, and fitted class indices in the order
of ``classes_``. A displayed coordinate belongs to its declared feature space:
original, affine-scaled, or transformed.

Exactness labels
================

``fitted`` and ``recorded`` identify estimator or recorder values. ``replayed``
and ``interpolated`` identify constructed states. Empirical evaluation along a
constructed path is mathematically valid but is not evidence of the estimator's
private optimization process.

Objectives and metrics
======================

MSE, MAE, :math:`R^2`, log-loss, accuracy, precision, and recall are reported
with their empirical roles. Regularization panels expose only supported public
hyperparameters and explicitly say when intercept penalties or private
normalization are not introspected.

Class labels
============

Indexed output is the mathematical default. ``show_class_labels=True`` adds
semantic labels for context; it does not invent meanings such as “accepted” or
“rejected.” Inspect ``LR-POLYNOMIAL`` and ``LOG-MULTI-FOCUS`` in the QA suite.

Multiclass logistic notation
============================

In a multiclass figure, :math:`K` is the number of fitted classes and
:math:`c_k=\texttt{classes\_}[k]` is the label in probability column :math:`k`.
The class index starts at zero: :math:`k=0,\ldots,K-1`. The number of input
features is :math:`d`; feature indices start at one: :math:`j=1,\ldots,d`.
Thus :math:`K=3,d=1` means three classes and one feature, not three features.
For every class, including those represented by vertical dots in a figure,
the same score and probability rules apply:

.. math::

   z_k(\mathbf{x})=\theta_{0,k}+\sum_{j=1}^{d}\theta_{j,k}x_j,
   \qquad
   \hat p(Y=c_k\mid\mathbf{x})=
   \frac{e^{z_k(\mathbf{x})}}{\sum_{\ell=0}^{K-1}e^{z_\ell(\mathbf{x})}}.

For a one-feature view, the score is
:math:`z_k=\theta_{0,k}+\theta_{1,k}x_1`. With two features it is
:math:`z_k=\theta_{0,k}+\theta_{1,k}x_1+\theta_{2,k}x_2`.
The class index and probability normalization stay the same. For example,
with :math:`K=3`, :math:`x_1=2`, biases :math:`(0.2,-0.4,0)`, and first-feature
weights :math:`(1,-0.5,0.25)`, the scores are :math:`(2.2,-1.4,0.5)` and
Softmax gives :math:`(0.8264,0.0226,0.1510)`. For class zero, the substitution
is :math:`z_0=0.2+1(2)=2.2` and
:math:`\hat p_0=e^{2.2}/(e^{2.2}+e^{-1.4}+e^{0.5})\approx0.8264`.
The other two classes use their own coefficient columns in the identical
formula. If :math:`x_2=1` and its coefficients are :math:`(0.1,0.4,-0.2)`,
the new scores are :math:`(2.3,-1.0,0.3)` and the probabilities are
:math:`(0.8531,0.0315,0.1155)`. This example is illustrative; animated
figures substitute their fitted model's coefficients instead.

A binary figure has :math:`K=2` and displays one independent logit with a
sigmoid for the positive class. A multiclass figure has :math:`K>2` and
displays one score per class, then uses Softmax or normalized one-vs-rest
sigmoids according to the estimator. Changing from the binary figure to a
three-class figure changes the output model as well as the notation.
