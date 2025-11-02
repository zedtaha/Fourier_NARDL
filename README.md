# 🌐 NARDL and Fourier NARDL Python Package
### by Dr. Taha Zaghdoudi  
**FSJEGJ, University of Jendouba – 2025**  
📜 *The code is open and free to use for research and academic purposes.*

---

## 📖 Overview

This repository implements the **Nonlinear Autoregressive Distributed Lag (NARDL)** and **Fourier NARDL** models in Python.  
It allows researchers to estimate both short- and long-run asymmetries between variables, with optional **Fourier terms** to capture smooth structural breaks and nonlinear dynamics.

The model automatically:
- Decomposes regressors into **positive** and **negative** partial sums,
- Performs **lag selection** using `AIC` or `BIC`,
- Computes **short-run** and **long-run** asymmetries (Wald tests),
- Runs **diagnostic tests** for residuals,
- Plots **dynamic and cumulative multipliers**,
- Performs **CUSUM** and **CUSUMSQ** stability tests.

---

## ⚙️ Installation

Ensure the following Python libraries are installed:

```bash
pip install numpy pandas scipy statsmodels matplotlib
import pandas as pd
import numpy as np
from nardl import NARDL

import pandas as pd
import numpy as np
from nardl import NARDL

# --- Simulated data ---
np.random.seed(42)
n = 150
x = np.cumsum(np.random.normal(0, 1, n))
z = np.cumsum(np.random.normal(0, 1, n))
y = 0.5 * x + 0.3 * z + np.random.normal(0, 1, n)

data = pd.DataFrame({'y': y, 'x': x, 'z': z})

# --- Estimate NARDL model ---
model = NARDL(formula='y ~ z | x', data=data, maxlag=3, ic='AIC', type='simple')
model.summary()

# --- Plot dynamic multipliers ---
model.plot_multipliers(variable='x')

# --- Plot CUSUM and CUSUMSQ stability tests ---
from nardl import plot_nardl
plot_nardl(model, which='both')
# --- Estimate Fourier NARDL model ---
model_fourier = NARDL(formula='y ~ z | x', data=data, maxlag=3, k=3, ic='AIC', type='fourier')
model_fourier.summary()

# --- Plot dynamic multipliers ---
model_fourier.plot_multipliers(variable='x')

# --- Plot stability tests ---
plot_nardl(model_fourier, which='cusumsq')


======================================================================
Fourier NARDL Model (Levels)
======================================================================
Selected Fourier frequency (k*): 2.370
Selected lags:
  p (Y lags): 2
  q (X lags): 1
  r (Z lags): {'z': 1}
Information Criterion: AIC = 245.3187

OLS Regression Results
----------------------------------------------------------------------
Dependent Variable: y
Observations: 150
R-squared: 0.781
Adj. R-squared: 0.764
F-statistic: 45.31 (p < 0.001)
----------------------------------------------------------------------
Coefficients:
Variable        Coef.      Std.Err      t-Stat      P>|t|
----------------------------------------------------------
y(-1)           0.5421     0.0911        5.95      0.000
x_pos(-1)       0.1283     0.0472        2.72      0.007
x_neg(-1)      -0.0561     0.0259       -2.17      0.032
z(-1)           0.2154     0.0802        2.69      0.008
Fourier_sin     0.0897     0.0398        2.25      0.026
Fourier_cos    -0.0721     0.0364       -1.98      0.049
Constant        0.0048     0.0024        1.96      0.052
======================================================================
Long-Run Asymmetry Estimates (Delta Method)
======================================================================
Variable: x
  Positive LR:  0.6234  (SE: 0.1453, t: 4.290, p: 0.0000) ***
  Negative LR: -0.2110  (SE: 0.0875, t: -2.410, p: 0.0173) *

Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05
======================================================================
Wald Tests for Asymmetry
======================================================================
Variable: x
  Short-run asymmetry (p-value): 0.0213
  Long-run asymmetry (p-value): 0.0048
======================================================================
Diagnostic Tests
======================================================================
Normality:
  Jarque-Bera: statistic = 1.3205, p-value = 0.5172
Serial Correlation:
  Breusch-Godfrey: statistic = 2.5184, p-value = 0.2831
Heteroskedasticity:
  Breusch-Pagan: statistic = 1.1298, p-value = 0.5697
ARCH Effect:
  ARCH(1): statistic = 0.9253, p-value = 0.3369
======================================================================
plot_nardl(model, which='both')   # both tests
plot_nardl(model, which='cusum')  # CUSUM only
plot_nardl(model, which='cusumsq')# CUSUMSQ only
---

## 🧩 References

- Shin, Y., Yu, B., & Greenwood-Nimmo, M. (2014).  
  *Modelling Asymmetric Cointegration and Dynamic Multipliers in a Nonlinear ARDL Framework.*  
  In R. Sickles & W. Horrace (Eds.), **Festschrift in Honor of Peter Schmidt**. Springer, New York, NY.  
  [https://doi.org/10.1007/978-1-4899-8008-3_9](https://doi.org/10.1007/978-1-4899-8008-3_9)

- Enders, W., & Jones, P. (2016).  
  *Granger Causality and Policy Analysis: A Cautionary Note on Using Fourier Approximations.*  
  *Oxford Bulletin of Economics and Statistics*, 78(5), 811–838.  
  [https://doi.org/10.1111/obes.12127](https://doi.org/10.1111/obes.12127)

- Zaghdoudi, T. (2025).  
  *NARDL and Fourier NARDL Python Implementation.*  
  Faculty of Legal, Economic and Management Sciences of Jendouba (FSJEGJ), Tunisia.

---

## ⚖️ License

This project is licensed under the terms of the **MIT License**.

MIT License

Copyright (c) 2025 Dr. Taha Zaghdoudi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

## ⭐ Citation

If you use this package in your academic work, please cite it as:

> Zaghdoudi, T. (2025). *NARDL and Fourier NARDL Python Implementation.*  
> Faculty of Legal, Economic and Management Sciences of Jendouba (FSJEGJ), Tunisia.  
> Available at: [https://github.com/tahazaghdoudi/nardl-python](https://github.com/tahazaghdoudi/nardl-python)

**BibTeX format:**
```bibtex
@software{zaghdoudi2025nardl,
  author       = {Taha Zaghdoudi},
  title        = {NARDL and Fourier NARDL Python Implementation},
  year         = {2025},
  institution  = {FSJEGJ, University of Jendouba},
  url          = {https://github.com/zedtaha/Fourier_NARDL},
  license      = {MIT}
}
