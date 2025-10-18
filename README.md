# 🧠 Machine Learning Regression Algorithms

A complete collection of **50 classical ML regression algorithms** (no Deep Learning).  
Each entry links to its code, import syntax, and mathematical formulation.

| S.No | Algorithm Name | Code | Import Syntax | Formula |
|------|----------------|------|----------------|----------|

### 🔹 Linear & Regularized Regression
| 1 | Simple Linear Regression | [Link](https://github.com/MainakVerse/Regressions/blob/main/simple-linear-regression.py) | `from sklearn.linear_model import LinearRegression` | ![eq](https://latex.codecogs.com/svg.image?\hat{y}=b_0+b_1x) |
| 2 | Multiple Linear Regression | [Link](https://github.com/MainakVerse/Regressions/blob/main/multiple-linear-regression.py) | `from sklearn.linear_model import LinearRegression` | ![eq](https://latex.codecogs.com/svg.image?\hat{y}=X\beta) |
| 3 | Ridge Regression | [Link](https://github.com/MainakVerse/Regressions/blob/main/ridge-regression.py) | `from sklearn.linear_model import Ridge` | ![eq](https://latex.codecogs.com/svg.image?\min_\beta\|y-X\beta\|^2+\lambda\|\beta\|^2) |
| 4 | RidgeCV | [Link](https://github.com/MainakVerse/Regressions/blob/main/ridgecv.py) | `from sklearn.linear_model import RidgeCV` | Same as Ridge |
| 5 | Lasso Regression | [Link](https://github.com/MainakVerse/Regressions/blob/main/lasso-regression.py) | `from sklearn.linear_model import Lasso` | ![eq](https://latex.codecogs.com/svg.image?\min_\beta\|y-X\beta\|^2+\lambda\|\beta\|_1) |
| 6 | LassoCV | [Link](https://github.com/MainakVerse/Regressions/blob/main/lassocv.py) | `from sklearn.linear_model import LassoCV` | Same as Lasso |
| 7 | Elastic Net Regression | [Link](https://github.com/MainakVerse/Regressions/blob/main/elastic-net.py) | `from sklearn.linear_model import ElasticNet` | ![eq](https://latex.codecogs.com/svg.image?\min_\beta\|y-X\beta\|^2+\lambda_1\|\beta\|_1+\lambda_2\|\beta\|^2) |
| 8 | ElasticNetCV | [Link](https://github.com/MainakVerse/Regressions/blob/main/elasticnetcv.py) | `from sklearn.linear_model import ElasticNetCV` | Same as Elastic Net |
| 9 | Bayesian Ridge Regression | [Link](https://github.com/MainakVerse/Regressions/blob/main/bayesian-ridge.py) | `from sklearn.linear_model import BayesianRidge` | ![eq](https://latex.codecogs.com/svg.image?p(\beta|X,y)\propto%20p(y|X,\beta)p(\beta)) |
| 10 | ARD Regression | [Link](https://github.com/MainakVerse/Regressions/blob/main/ard-regression.py) | `from sklearn.linear_model import ARDRegression` | Bayesian relevance weights |
| 11 | Polynomial Regression | [Link](https://github.com/MainakVerse/Regressions/blob/main/polynomial-regression.py) | `from sklearn.preprocessing import PolynomialFeatures` | ![eq](https://latex.codecogs.com/svg.image?\hat{y}=b_0+b_1x+b_2x^2+...+b_nx^n) |
| 12 | Least Angle Regression (LARS) | [Link](https://github.com/MainakVerse/Regressions/blob/main/lars.py) | `from sklearn.linear_model import Lars` | Iterative feature inclusion |
| 13 | LassoLars | [Link](https://github.com/MainakVerse/Regressions/blob/main/lasso-lars.py) | `from sklearn.linear_model import LassoLars` | LARS with L1 constraint |
| 14 | LassoLarsCV | [Link](https://github.com/MainakVerse/Regressions/blob/main/lassolarscv.py) | `from sklearn.linear_model import LassoLarsCV` | Cross-validated |
| 15 | LassoLarsIC | [Link](https://github.com/MainakVerse/Regressions/blob/main/lassolarsic.py) | `from sklearn.linear_model import LassoLarsIC` | Info criterion (AIC/BIC) |
| 16 | Orthogonal Matching Pursuit (OMP) | [Link](https://github.com/MainakVerse/Regressions/blob/main/omp.py) | `from sklearn.linear_model import OrthogonalMatchingPursuit` | Greedy feature selection |
| 17 | OrthogonalMatchingPursuitCV | [Link](https://github.com/MainakVerse/Regressions/blob/main/ompcv.py) | `from sklearn.linear_model import OrthogonalMatchingPursuitCV` | Cross-validated OMP |

### 🔹 Robust & Outlier-Resistant Regression
| 18 | Huber Regression | [Link](https://github.com/MainakVerse/Regressions/blob/main/huber.py) | `from sklearn.linear_model import HuberRegressor` | ![eq](https://latex.codecogs.com/svg.image?\min_\beta\sum%20L_\delta(y_i-\hat{y}_i)) |
| 19 | Theil–Sen Regression | [Link](https://github.com/MainakVerse/Regressions/blob/main/theilsen.py) | `from sklearn.linear_model import TheilSenRegressor` | Median of pairwise slopes |
| 20 | RANSAC Regression | [Link](https://github.com/MainakVerse/Regressions/blob/main/ransac.py) | `from sklearn.linear_model import RANSACRegressor` | Iterative inlier fitting |
| 21 | Quantile Regression | [Link](https://github.com/MainakVerse/Regressions/blob/main/quantile.py) | `from sklearn.linear_model import QuantileRegressor` | ![eq](https://latex.codecogs.com/svg.image?\min_\beta\sum\rho_\tau(y_i-x_i\beta)) |
| 22 | Robust Linear Model (RLM) | [Link](https://github.com/MainakVerse/Regressions/blob/main/rlm.py) | `from statsmodels.robust.robust_linear_model import RLM` | ![eq](https://latex.codecogs.com/svg.image?\min_\beta\sum\rho(y_i-x_i\beta)) |

### 🔹 Support Vector & Kernel-Based
| 23 | Support Vector Regression (SVR) | [Link](https://github.com/MainakVerse/Regressions/blob/main/svr.py) | `from sklearn.svm import SVR` | ![eq](https://latex.codecogs.com/svg.image?\min\frac{1}{2}\|w\|^2+C\sum\xi_i) |
| 24 | LinearSVR | [Link](https://github.com/MainakVerse/Regressions/blob/main/linear-svr.py) | `from sklearn.svm import LinearSVR` | Linear ε-insensitive loss |
| 25 | Kernel Ridge Regression | [Link](https://github.com/MainakVerse/Regressions/blob/main/kernel-ridge.py) | `from sklearn.kernel_ridge import KernelRidge` | ![eq](https://latex.codecogs.com/svg.image?\alpha=(K+\lambda%20I)^{-1}y) |

### 🔹 Tree-Based & Ensemble Regression
| 26 | Decision Tree Regressor | [Link](https://github.com/MainakVerse/Regressions/blob/main/decision-tree.py) | `from sklearn.tree import DecisionTreeRegressor` | ![eq](https://latex.codecogs.com/svg.image?\min\sum(y_i-\bar{y}_j)^2) |
| 27 | Extra Trees Regressor | [Link](https://github.com/MainakVerse/Regressions/blob/main/extra-trees.py) | `from sklearn.ensemble import ExtraTreesRegressor` | Randomized tree splits |
| 28 | Random Forest Regressor | [Link](https://github.com/MainakVerse/Regressions/blob/main/random-forest.py) | `from sklearn.ensemble import RandomForestRegressor` | ![eq](https://latex.codecogs.com/svg.image?\hat{y}=\frac{1}{T}\sum_{t=1}^T\hat{y}_t) |
| 29 | Bagging Regressor | [Link](https://github.com/MainakVerse/Regressions/blob/main/bagging.py) | `from sklearn.ensemble import BaggingRegressor` | Averaged bootstrap models |
| 30 | AdaBoost Regressor | [Link](https://github.com/MainakVerse/Regressions/blob/main/adaboost.py) | `from sklearn.ensemble import AdaBoostRegressor` | ![eq](https://latex.codecogs.com/svg.image?F_M(x)=\sum_{m=1}^M\alpha_mh_m(x)) |
| 31 | Gradient Boosting Regressor | [Link](https://github.com/MainakVerse/Regressions/blob/main/gradient-boosting.py) | `from sklearn.ensemble import GradientBoostingRegressor` | Fit residuals iteratively |
| 32 | Histogram-Based GBR | [Link](https://github.com/MainakVerse/Regressions/blob/main/hist-gradient-boosting.py) | `from sklearn.ensemble import HistGradientBoostingRegressor` | Histogram optimization |
| 33 | XGBoost Regressor | [Link](https://github.com/MainakVerse/Regressions/blob/main/xgboost.py) | `from xgboost import XGBRegressor` | Optimized GBDT |
| 34 | LightGBM Regressor | [Link](https://github.com/MainakVerse/Regressions/blob/main/lightgbm.py) | `from lightgbm import LGBMRegressor` | Leaf-wise boosting |
| 35 | CatBoost Regressor | [Link](https://github.com/MainakVerse/Regressions/blob/main/catboost.py) | `from catboost import CatBoostRegressor` | Ordered boosting |

### 🔹 Instance-Based & Lazy Learning
| 36 | K-Nearest Neighbors (KNN) | [Link](https://github.com/MainakVerse/Regressions/blob/main/knn.py) | `from sklearn.neighbors import KNeighborsRegressor` | ![eq](https://latex.codecogs.com/svg.image?\hat{y}=\frac{1}{k}\sum_{i\in N_k(x)}y_i) |
| 37 | Radius Neighbors Regressor | [Link](https://github.com/MainakVerse/Regressions/blob/main/radius-neighbors.py) | `from sklearn.neighbors import RadiusNeighborsRegressor` | Weighted neighbors |
| 38 | Locally Weighted Regression (LWR) | [Link](https://github.com/MainakVerse/Regressions/blob/main/lwr.py) | Manual | ![eq](https://latex.codecogs.com/svg.image?\hat{y}(x)=w(x)^T(X^TX)^{-1}X^Ty) |

### 🔹 Probabilistic & Bayesian Models
| 39 | Gaussian Process Regressor | [Link](https://github.com/MainakVerse/Regressions/blob/main/gaussian-process.py) | `from sklearn.gaussian_process import GaussianProcessRegressor` | ![eq](https://latex.codecogs.com/svg.image?y\sim\mathcal{N}(0,K+\sigma^2I)) |
| 40 | Generalized Linear Model (GLM) | [Link](https://github.com/MainakVerse/Regressions/blob/main/glm.py) | `from statsmodels.genmod.generalized_linear_model import GLM` | ![eq](https://latex.codecogs.com/svg.image?g(\mu)=X\beta) |
| 41 | Poisson Regression | [Link](https://github.com/MainakVerse/Regressions/blob/main/poisson.py) | `from sklearn.linear_model import PoissonRegressor` | ![eq](https://latex.codecogs.com/svg.image?\log(\lambda)=X\beta) |
| 42 | Gamma Regression | [Link](https://github.com/MainakVerse/Regressions/blob/main/gamma.py) | `from sklearn.linear_model import GammaRegressor` | ![eq](https://latex.codecogs.com/svg.image?E[y]=\exp(X\beta)) |
| 43 | Tweedie Regressor | [Link](https://github.com/MainakVerse/Regressions/blob/main/tweedie.py) | `from sklearn.linear_model import TweedieRegressor` | ![eq](https://latex.codecogs.com/svg.image?Var(y)=\phi\mu^p) |

### 🔹 Meta & Hybrid Models
| 44 | Stacking Regressor | [Link](https://github.com/MainakVerse/Regressions/blob/main/stacking.py) | `from sklearn.ensemble import StackingRegressor` | ![eq](https://latex.codecogs.com/svg.image?\hat{y}=f(g_1(x),...,g_m(x))) |
| 45 | Voting Regressor | [Link](https://github.com/MainakVerse/Regressions/blob/main/voting.py) | `from sklearn.ensemble import VotingRegressor` | Averaged predictions |
| 46 | Averaging Regressor | [Link](https://github.com/MainakVerse/Regressions/blob/main/averaging.py) | Manual | Simple mean |
| 47 | Blending Regressor | [Link](https://github.com/MainakVerse/Regressions/blob/main/blending.py) | Manual | Weighted ensemble |

### 🔹 Specialized & Statistical Variants
| 48 | Isotonic Regression | [Link](https://github.com/MainakVerse/Regressions/blob/main/isotonic.py) | `from sklearn.isotonic import IsotonicRegression` | Monotonic constraint |
| 49 | Principal Component Regression (PCR) | [Link](https://github.com/MainakVerse/Regressions/blob/main/pcr.py) | Combine PCA + LinearRegression | ![eq](https://latex.codecogs.com/svg.image?\hat{y}=Z\beta) |
| 50 | Partial Least Squares (PLS) Regression | [Link](https://github.com/MainakVerse/Regressions/blob/main/pls.py) | `from sklearn.cross_decomposition import PLSRegression` | ![eq](https://latex.codecogs.com/svg.image?\max\text{Cov}(Xw,Yc)) |
