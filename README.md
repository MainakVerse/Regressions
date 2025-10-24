# 🧠 Machine Learning Regression Algorithms

A complete collection of **50 classical Machine Learning regression algorithms** (no Deep Learning).  
Each algorithm includes its name, code file link, and corresponding import syntax.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-≥1.0-orange?logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-success)
![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-brightgreen)

---

## 🔹 Linear & Regularized Regression

| S.No | Algorithm | Code | Import Syntax |
|------|------------|------|----------------|
| 1 | Simple Linear Regression | [Link](https://github.com/MainakVerse/Regressions/blob/main/simple-linear-regression.py) | `from sklearn.linear_model import LinearRegression` |
| 2 | Multiple Linear Regression | [Link](https://github.com/MainakVerse/Regressions/blob/main/multiple-linear-regression.py) | `from sklearn.linear_model import LinearRegression` |
| 3 | Ridge Regression | [Link](https://github.com/MainakVerse/Regressions/blob/main/ridge-regression.py) | `from sklearn.linear_model import Ridge` |
| 4 | RidgeCV | [Link](https://github.com/MainakVerse/Regressions/blob/main/ridge-cross-validation.py) | `from sklearn.linear_model import RidgeCV` |
| 5 | Lasso Regression | [Link](https://github.com/MainakVerse/Regressions/blob/main/lasso-regression.py) | `from sklearn.linear_model import Lasso` |
| 6 | LassoCV | [Link](https://github.com/MainakVerse/Regressions/blob/main/lassocv.py) | `from sklearn.linear_model import LassoCV` |
| 7 | Elastic Net Regression | [Link](https://github.com/MainakVerse/Regressions/blob/main/elastic-net.py) | `from sklearn.linear_model import ElasticNet` |
| 8 | ElasticNetCV | [Link](https://github.com/MainakVerse/Regressions/blob/main/elasticnetcv.py) | `from sklearn.linear_model import ElasticNetCV` |
| 9 | Bayesian Ridge Regression | [Link](https://github.com/MainakVerse/Regressions/blob/main/bayesian-ridge.py) | `from sklearn.linear_model import BayesianRidge` |
| 10 | ARD Regression | [Link](https://github.com/MainakVerse/Regressions/blob/main/ard-regression.py) | `from sklearn.linear_model import ARDRegression` |
| 11 | Polynomial Regression | [Link](https://github.com/MainakVerse/Regressions/blob/main/polynomial-regression.py) | `from sklearn.preprocessing import PolynomialFeatures` |
| 12 | Least Angle Regression (LARS) | [Link](https://github.com/MainakVerse/Regressions/blob/main/lars.py) | `from sklearn.linear_model import Lars` |
| 13 | LassoLars | [Link](https://github.com/MainakVerse/Regressions/blob/main/lasso-lars.py) | `from sklearn.linear_model import LassoLars` |
| 14 | LassoLarsCV | [Link](https://github.com/MainakVerse/Regressions/blob/main/lassolarscv.py) | `from sklearn.linear_model import LassoLarsCV` |
| 15 | LassoLarsIC | [Link](https://github.com/MainakVerse/Regressions/blob/main/lassolarsic.py) | `from sklearn.linear_model import LassoLarsIC` |
| 16 | Orthogonal Matching Pursuit (OMP) | [Link](https://github.com/MainakVerse/Regressions/blob/main/omp.py) | `from sklearn.linear_model import OrthogonalMatchingPursuit` |
| 17 | OrthogonalMatchingPursuitCV | [Link](https://github.com/MainakVerse/Regressions/blob/main/ompcv.py) | `from sklearn.linear_model import OrthogonalMatchingPursuitCV` |

---

## 🔹 Robust & Outlier-Resistant Regression

| S.No | Algorithm | Code | Import Syntax |
|------|------------|------|----------------|
| 18 | Huber Regression | [Link](https://github.com/MainakVerse/Regressions/blob/main/huber.py) | `from sklearn.linear_model import HuberRegressor` |
| 19 | Theil–Sen Regression | [Link](https://github.com/MainakVerse/Regressions/blob/main/theilsen.py) | `from sklearn.linear_model import TheilSenRegressor` |
| 20 | RANSAC Regression | [Link](https://github.com/MainakVerse/Regressions/blob/main/ransac.py) | `from sklearn.linear_model import RANSACRegressor` |
| 21 | Quantile Regression | [Link](https://github.com/MainakVerse/Regressions/blob/main/quantile.py) | `from sklearn.linear_model import QuantileRegressor` |
| 22 | Robust Linear Model (RLM) | [Link](https://github.com/MainakVerse/Regressions/blob/main/rlm.py) | `from statsmodels.robust.robust_linear_model import RLM` |

---

## 🔹 Support Vector & Kernel-Based

| S.No | Algorithm | Code | Import Syntax |
|------|------------|------|----------------|
| 23 | Support Vector Regression (SVR) | [Link](https://github.com/MainakVerse/Regressions/blob/main/svr.py) | `from sklearn.svm import SVR` |
| 24 | LinearSVR | [Link](https://github.com/MainakVerse/Regressions/blob/main/linear-svr.py) | `from sklearn.svm import LinearSVR` |
| 25 | Kernel Ridge Regression | [Link](https://github.com/MainakVerse/Regressions/blob/main/kernel-ridge.py) | `from sklearn.kernel_ridge import KernelRidge` |

---

## 🔹 Tree-Based & Ensemble Regression

| S.No | Algorithm | Code | Import Syntax |
|------|------------|------|----------------|
| 26 | Decision Tree Regressor | [Link](https://github.com/MainakVerse/Regressions/blob/main/decision-tree.py) | `from sklearn.tree import DecisionTreeRegressor` |
| 27 | Extra Trees Regressor | [Link](https://github.com/MainakVerse/Regressions/blob/main/extra-trees.py) | `from sklearn.ensemble import ExtraTreesRegressor` |
| 28 | Random Forest Regressor | [Link](https://github.com/MainakVerse/Regressions/blob/main/random-forest.py) | `from sklearn.ensemble import RandomForestRegressor` |
| 29 | Bagging Regressor | [Link](https://github.com/MainakVerse/Regressions/blob/main/bagging.py) | `from sklearn.ensemble import BaggingRegressor` |
| 30 | AdaBoost Regressor | [Link](https://github.com/MainakVerse/Regressions/blob/main/adaboost.py) | `from sklearn.ensemble import AdaBoostRegressor` |
| 31 | Gradient Boosting Regressor | [Link](https://github.com/MainakVerse/Regressions/blob/main/gradient-boosting.py) | `from sklearn.ensemble import GradientBoostingRegressor` |
| 32 | Histogram-Based GBR | [Link](https://github.com/MainakVerse/Regressions/blob/main/hist-gradient-boosting.py) | `from sklearn.ensemble import HistGradientBoostingRegressor` |
| 33 | XGBoost Regressor | [Link](https://github.com/MainakVerse/Regressions/blob/main/xgboost.py) | `from xgboost import XGBRegressor` |
| 34 | LightGBM Regressor | [Link](https://github.com/MainakVerse/Regressions/blob/main/lightgbm.py) | `from lightgbm import LGBMRegressor` |
| 35 | CatBoost Regressor | [Link](https://github.com/MainakVerse/Regressions/blob/main/catboost.py) | `from catboost import CatBoostRegressor` |

---

## 🔹 Instance-Based & Lazy Learning

| S.No | Algorithm | Code | Import Syntax |
|------|------------|------|----------------|
| 36 | K-Nearest Neighbors (KNN) | [Link](https://github.com/MainakVerse/Regressions/blob/main/knn.py) | `from sklearn.neighbors import KNeighborsRegressor` |
| 37 | Radius Neighbors Regressor | [Link](https://github.com/MainakVerse/Regressions/blob/main/radius-neighbors.py) | `from sklearn.neighbors import RadiusNeighborsRegressor` |
| 38 | Locally Weighted Regression (LWR) | [Link](https://github.com/MainakVerse/Regressions/blob/main/lwr.py) | Manual |

---

## 🔹 Probabilistic & Bayesian Models

| S.No | Algorithm | Code | Import Syntax |
|------|------------|------|----------------|
| 39 | Gaussian Process Regressor | [Link](https://github.com/MainakVerse/Regressions/blob/main/gaussian-process.py) | `from sklearn.gaussian_process import GaussianProcessRegressor` |
| 40 | Generalized Linear Model (GLM) | [Link](https://github.com/MainakVerse/Regressions/blob/main/glm.py) | `from statsmodels.genmod.generalized_linear_model import GLM` |
| 41 | Poisson Regression | [Link](https://github.com/MainakVerse/Regressions/blob/main/poisson.py) | `from sklearn.linear_model import PoissonRegressor` |
| 42 | Gamma Regression | [Link](https://github.com/MainakVerse/Regressions/blob/main/gamma.py) | `from sklearn.linear_model import GammaRegressor` |
| 43 | Tweedie Regressor | [Link](https://github.com/MainakVerse/Regressions/blob/main/tweedie.py) | `from sklearn.linear_model import TweedieRegressor` |

---

## 🔹 Meta & Hybrid Models

| S.No | Algorithm | Code | Import Syntax |
|------|------------|------|----------------|
| 44 | Stacking Regressor | [Link](https://github.com/MainakVerse/Regressions/blob/main/stacking.py) | `from sklearn.ensemble import StackingRegressor` |
| 45 | Voting Regressor | [Link](https://github.com/MainakVerse/Regressions/blob/main/voting.py) | `from sklearn.ensemble import VotingRegressor` |
| 46 | Averaging Regressor | [Link](https://github.com/MainakVerse/Regressions/blob/main/averaging.py) | Manual |
| 47 | Blending Regressor | [Link](https://github.com/MainakVerse/Regressions/blob/main/blending.py) | Manual |

---

## 🔹 Specialized & Statistical Variants

| S.No | Algorithm | Code | Import Syntax |
|------|------------|------|----------------|
| 48 | Isotonic Regression | [Link](https://github.com/MainakVerse/Regressions/blob/main/isotonic.py) | `from sklearn.isotonic import IsotonicRegression` |
| 49 | Principal Component Regression (PCR) | [Link](https://github.com/MainakVerse/Regressions/blob/main/pcr.py) | Combine PCA + LinearRegression |
| 50 | Partial Least Squares (PLS) Regression | [Link](https://github.com/MainakVerse/Regressions/blob/main/pls.py) | `from sklearn.cross_decomposition import PLSRegression` |
