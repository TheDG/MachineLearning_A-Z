# R-Squared
- SS<sub>res</sub>  = SUM(y<sub>i</sub> - y<sup>^</sup><sub>i</sub>)<sup>2</sup>
- SS<sub>tot</sub>  = SUM(y<sub>i</sub> - y<sub>avg</sub>)<sup>2</sup>
- R<sup>2</sup> = 1 - SS<sub> res</sub> / SS <sub>tot </sub>

# Adjusted R-Squared
- Penalizes that addition of variables

- Adj R<sup>2</sup> = 1 - R<sup>2</sup>((n-1/(n-p-1)))
  - n = sample test_size
  - p = # of regresors

#### For script for backward elimination formula using Adjusted R-Squared Check Section 5 / Multiple Linear Regresion Models

# [Model Pro - Con Cheat sheet](https://sds-platform-private.s3-us-east-2.amazonaws.com/uploads/P14-Regression-Pros-Cons.pdf)

# How do I know which model to choose for my problem ?

First, you need to figure out whether your problem is linear or non linear. You will learn how to do that in Part 10 - Model Selection. Then:

If your problem is linear, you should go for Simple Linear Regression if you only have one feature, and Multiple Linear Regression if you have several features.

If your problem is non linear, you should go for Polynomial Regression, SVR, Decision Tree or Random Forest. Then which one should you choose among these four ? That you will learn in Part 10 - Model Selection.
