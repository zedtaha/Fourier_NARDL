#******************************************************************
# NARDL and FOURIER NARDL FUNCTION by Dr. Taha Zaghdoudi
# by Dr. Taha Zaghdoudi
#  FSJEGJ, University of Jendouba 2025
# The code is open and free to use
#*****************************************************************
import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import inv
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.diagnostic import het_breuschpagan, acorr_breusch_godfrey
from statsmodels.stats.stattools import jarque_bera
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class NARDL:
    def __init__(self, formula, data, maxlag=4, k=3, ic='AIC', type='simple'):
        """
        NARDL Model Estimation
        
        Parameters:
        -----------
        formula : str
            Formula in format "Y ~ Z1 + Z2 | X" where Y is dependent,
            Z variables are controls, X is variable to decompose
        data : pd.DataFrame
            Input data
        maxlag : int
            Maximum lag length to search
        k : float
            Maximum Fourier frequency
        ic : str
            Information criterion ('AIC' or 'BIC')
        type : str
            Model type ('simple' or 'fourier')
        """
        self.formula = formula
        self.data = data.copy()
        self.maxlag = maxlag
        self.k = k
        self.ic = ic.upper()
        self.type = type
        
        # Parse formula
        self._parse_formula()
        
        # Decompose variables
        self._decompose_variables()
        
        # Fit model
        self._fit_model()
        
        # Compute long-run multipliers
        self._compute_long_run()
        
        # Wald tests
        self._wald_tests()
        
        # Diagnostic tests
        self._diagnostic_tests()
        
        # Dynamic multipliers
        self._compute_dynamic_multipliers()
    
    def _parse_formula(self):
        """Parse the formula to extract variable names"""
        parts = self.formula.split('~')
        self.yname = parts[0].strip()
        
        rhs = parts[1].split('|')
        self.znames = [x.strip() for x in rhs[0].split('+')]  # Control variables
        self.xnames = [x.strip() for x in rhs[1].split('+')]  # Variables to decompose
    
    def _decompose_variables(self):
        """Decompose X variables into positive and negative partial sums"""
        for var in self.xnames:
            dx = self.data[var].diff()
            self.data[f'{var}_pos'] = np.concatenate([[0], np.maximum(dx[1:], 0).cumsum()])
            self.data[f'{var}_neg'] = np.concatenate([[0], np.minimum(dx[1:], 0).cumsum()])
    
    def _fit_model(self):
        """Grid search over lag combinations to find best model"""
        n = len(self.data)
        
        # Grid search parameters
        if self.type == 'fourier':
            pramk = np.arange(0.1, self.k + 0.01, 0.01)
        else:
            pramk = [0]
        
        best_ic = np.inf
        best_model = None
        best_lags = None
        best_kstar = 0 if self.type == 'simple' else np.nan
        
        # Create all r combinations
        r_combinations = self._get_r_combinations()
        
        print(f"Searching over {len(pramk)} Fourier frequencies, {self.maxlag} p lags, "
              f"{self.maxlag+1} q lags, and {len(r_combinations)} r combinations...")
        
        total_models = 0
        for kstar in pramk:
            t = np.arange(n)
            
            # Add Fourier terms
            if self.type == 'fourier':
                self.data['sin_t'] = np.sin(2 * np.pi * kstar * t / n)
                self.data['cos_t'] = np.cos(2 * np.pi * kstar * t / n)
            
            for p in range(1, self.maxlag + 1):
                for q in range(0, self.maxlag + 1):
                    for r_vals in r_combinations:
                        total_models += 1
                        
                        # Build model data
                        model_data = self._build_model_data(p, q, r_vals)
                        
                        if model_data is None:
                            continue
                        
                        # Fit model
                        try:
                            formula_str = self._build_formula(p, q, r_vals)
                            model = ols(formula_str, data=model_data).fit()
                            
                            # Calculate IC
                            nobs = model.nobs
                            k_params = len(model.params)
                            ssr = model.ssr
                            
                            if self.ic == 'AIC':
                                ic_val = nobs * np.log(ssr / nobs) + 2 * k_params
                            else:  # BIC
                                ic_val = nobs * np.log(ssr / nobs) + k_params * np.log(nobs)
                            
                            if ic_val < best_ic:
                                best_ic = ic_val
                                best_model = model
                                best_lags = {'p': p, 'q': q, 'r': dict(zip(self.znames, r_vals))}
                                best_kstar = kstar
                                
                        except:
                            continue
        
        print(f"Evaluated {total_models} models. Best {self.ic}: {best_ic:.4f}")
        
        if best_model is None:
            raise ValueError("No valid models found. Check your data.")
        
        self.model = best_model
        self.best_lags = best_lags
        self.best_kstar = best_kstar
        self.best_ic = best_ic
    
    def _get_r_combinations(self):
        """Generate all combinations of r lags for control variables"""
        from itertools import product
        r_ranges = [range(0, self.maxlag + 1) for _ in self.znames]
        return list(product(*r_ranges))
    
    def _build_model_data(self, p, q, r_vals):
        """Build dataset with all lags for given p, q, r"""
        df = self.data.copy()
        
        # Dependent variable
        df['Y'] = df[self.yname]
        
        # Lagged Y
        for j in range(1, p + 1):
            df[f'Y_L{j}'] = df['Y'].shift(j)
        
        # Decomposed X variables
        for var in self.xnames:
            # Contemporary
            df[f'{var}_pos_L0'] = df[f'{var}_pos']
            df[f'{var}_neg_L0'] = df[f'{var}_neg']
            
            # Lagged
            for j in range(1, q + 1):
                df[f'{var}_pos_L{j}'] = df[f'{var}_pos'].shift(j)
                df[f'{var}_neg_L{j}'] = df[f'{var}_neg'].shift(j)
        
        # Control variables Z
        for idx, var in enumerate(self.znames):
            r_x = r_vals[idx]
            
            # Contemporary
            df[f'{var}_L0'] = df[var]
            
            # Lagged
            for j in range(1, r_x + 1):
                df[f'{var}_L{j}'] = df[var].shift(j)
        
        # Drop NaN rows
        df = df.dropna()
        
        if len(df) < 30:  # Minimum sample size
            return None
        
        return df
    
    def _build_formula(self, p, q, r_vals):
        """Build regression formula string"""
        rhs = []
        
        # Lagged Y
        for j in range(1, p + 1):
            rhs.append(f'Y_L{j}')
        
        # Decomposed X
        for var in self.xnames:
            rhs.append(f'{var}_pos_L0')
            rhs.append(f'{var}_neg_L0')
            
            for j in range(1, q + 1):
                rhs.append(f'{var}_pos_L{j}')
                rhs.append(f'{var}_neg_L{j}')
        
        # Control variables Z
        for idx, var in enumerate(self.znames):
            r_x = r_vals[idx]
            rhs.append(f'{var}_L0')
            
            for j in range(1, r_x + 1):
                rhs.append(f'{var}_L{j}')
        
        # Fourier terms
        if self.type == 'fourier':
            rhs.extend(['sin_t', 'cos_t'])
        
        return f"Y ~ {' + '.join(rhs)}"
    
    def _compute_long_run(self):
        """Compute long-run multipliers using delta method"""
        coefs = self.model.params
        vcov = self.model.cov_params()
        
        # AR coefficients
        y_lag_names = [f'Y_L{j}' for j in range(1, self.best_lags['p'] + 1)]
        y_lag_sum = sum(coefs[name] for name in y_lag_names if name in coefs)
        denominator = 1 - y_lag_sum
        
        self.long_run = {}
        self.long_run_se = {}
        
        for var in self.xnames:
            # Get coefficient names
            pos_names = [f'{var}_pos_L0']
            neg_names = [f'{var}_neg_L0']
            
            for j in range(1, self.best_lags['q'] + 1):
                pos_names.append(f'{var}_pos_L{j}')
                neg_names.append(f'{var}_neg_L{j}')
            
            # Sum coefficients
            sum_pos = sum(coefs[name] for name in pos_names if name in coefs)
            sum_neg = sum(coefs[name] for name in neg_names if name in coefs)
            
            # Long-run multipliers
            lr_pos = sum_pos / denominator
            lr_neg = sum_neg / denominator
            
            # Delta method for standard errors
            all_names = coefs.index.tolist()
            n_coef = len(all_names)
            
            # Gradient for positive
            grad_pos = np.zeros(n_coef)
            for name in pos_names:
                if name in all_names:
                    grad_pos[all_names.index(name)] = 1 / denominator
            
            for name in y_lag_names:
                if name in all_names:
                    grad_pos[all_names.index(name)] = sum_pos / (denominator ** 2)
            
            # Gradient for negative
            grad_neg = np.zeros(n_coef)
            for name in neg_names:
                if name in all_names:
                    grad_neg[all_names.index(name)] = 1 / denominator
            
            for name in y_lag_names:
                if name in all_names:
                    grad_neg[all_names.index(name)] = sum_neg / (denominator ** 2)
            
            # Compute variance
            var_lr_pos = grad_pos @ vcov.values @ grad_pos
            var_lr_neg = grad_neg @ vcov.values @ grad_neg
            
            se_lr_pos = np.sqrt(var_lr_pos)
            se_lr_neg = np.sqrt(var_lr_neg)
            
            self.long_run[var] = {'positive': lr_pos, 'negative': lr_neg}
            self.long_run_se[var] = {'positive': se_lr_pos, 'negative': se_lr_neg}
    
    def _wald_tests(self):
        """Perform Wald tests for asymmetry"""
        self.wald = {}
        
        for var in self.xnames:
            pos_names = [f'{var}_pos_L0']
            neg_names = [f'{var}_neg_L0']
            
            for j in range(1, self.best_lags['q'] + 1):
                pos_names.append(f'{var}_pos_L{j}')
                neg_names.append(f'{var}_neg_L{j}')
            
            # Short-run test: θ₀⁺ = θ₀⁻
            short_run_test = None
            try:
                hyp = f'{var}_pos_L0 = {var}_neg_L0'
                short_run_test = self.model.f_test(hyp)
            except:
                pass
            
            # Long-run test: sum(θ⁺) = sum(θ⁻)
            long_run_test = None
            try:
                hyp = ' + '.join(pos_names) + ' = ' + ' + '.join(neg_names)
                long_run_test = self.model.f_test(hyp)
            except:
                pass
            
            self.wald[var] = {
                'short_run': short_run_test,
                'long_run': long_run_test
            }
    
    def _diagnostic_tests(self):
        """Perform diagnostic tests"""
        resid = self.model.resid
        
        self.diagnostics = {}
        
        # Jarque-Bera test
        try:
            jb_stat, jb_pval, _, _ = jarque_bera(resid)
            self.diagnostics['jarque_bera'] = {'statistic': jb_stat, 'pvalue': jb_pval}
        except:
            self.diagnostics['jarque_bera'] = None
        
        # Shapiro-Wilk test
        try:
            sw_stat, sw_pval = stats.shapiro(resid)
            self.diagnostics['shapiro_wilk'] = {'statistic': sw_stat, 'pvalue': sw_pval}
        except:
            self.diagnostics['shapiro_wilk'] = None
        
        # Breusch-Godfrey test
        try:
            bg = acorr_breusch_godfrey(self.model, nlags=self.best_lags['p'])
            self.diagnostics['breusch_godfrey'] = {
                'statistic': bg[0], 
                'pvalue': bg[1]
            }
        except:
            self.diagnostics['breusch_godfrey'] = None
        
        # Breusch-Pagan test
        try:
            bp = het_breuschpagan(resid, self.model.model.exog)
            self.diagnostics['breusch_pagan'] = {
                'statistic': bp[0], 
                'pvalue': bp[1]
            }
        except:
            self.diagnostics['breusch_pagan'] = None
        
        # ARCH test
        try:
            resid_sq = resid ** 2
            arch_model = sm.OLS(resid_sq[1:], sm.add_constant(resid_sq[:-1])).fit()
            arch_stat = arch_model.rsquared * len(resid_sq)
            arch_pval = 1 - stats.chi2.cdf(arch_stat, 1)
            self.diagnostics['arch'] = {
                'statistic': arch_stat, 
                'pvalue': arch_pval
            }
        except:
            self.diagnostics['arch'] = None
    
    def _compute_dynamic_multipliers(self, horizon=None):
        """Compute dynamic and cumulative multipliers"""
        if horizon is None:
            horizon = max(self.best_lags['p'], self.best_lags['q']) + 10
        
        coefs = self.model.params
        
        # AR coefficients
        y_lag_names = [f'Y_L{j}' for j in range(1, self.best_lags['p'] + 1)]
        phi_coefs = np.array([coefs[name] if name in coefs else 0 
                               for name in y_lag_names])
        
        self.dynamic_multipliers = {}
        
        for var in self.xnames:
            # Get X coefficients
            theta_pos = np.zeros(self.best_lags['q'] + 1)
            theta_neg = np.zeros(self.best_lags['q'] + 1)
            
            for j in range(self.best_lags['q'] + 1):
                name = f'{var}_pos_L{j}' if j > 0 else f'{var}_pos_L0'
                if name in coefs:
                    theta_pos[j] = coefs[name]
                
                name = f'{var}_neg_L{j}' if j > 0 else f'{var}_neg_L0'
                if name in coefs:
                    theta_neg[j] = coefs[name]
            
            # Compute impulse responses
            mult_pos = np.zeros(horizon)
            mult_neg = np.zeros(horizon)
            
            # Initialize
            mult_pos[0] = theta_pos[0]
            mult_neg[0] = theta_neg[0]
            
            # Recursive computation
            for h in range(1, horizon):
                # Effect from lagged Y
                for j in range(min(h, self.best_lags['p'])):
                    mult_pos[h] += phi_coefs[j] * mult_pos[h - j - 1]
                    mult_neg[h] += phi_coefs[j] * mult_neg[h - j - 1]
                
                # Direct effect from X
                if h < len(theta_pos):
                    mult_pos[h] += theta_pos[h]
                    mult_neg[h] += theta_neg[h]
            
            # Cumulative multipliers
            cum_mult_pos = np.cumsum(mult_pos)
            cum_mult_neg = np.cumsum(mult_neg)
            
            self.dynamic_multipliers[var] = {
                'positive': {
                    'multiplier': mult_pos,
                    'cumulative': cum_mult_pos,
                    'horizon': np.arange(horizon)
                },
                'negative': {
                    'multiplier': mult_neg,
                    'cumulative': cum_mult_neg,
                    'horizon': np.arange(horizon)
                }
            }
    
    def summary(self):
        """Print model summary"""
        print("=" * 70)
        print(f"{'Fourier ' if self.type == 'fourier' else ''}NARDL Model (Levels)")
        print("=" * 70)
        
        if self.type == 'fourier':
            print(f"Selected Fourier frequency (k*): {self.best_kstar:.3f}")
        
        print(f"Selected lags:")
        print(f"  p (Y lags): {self.best_lags['p']}")
        print(f"  q (X lags): {self.best_lags['q']}")
        print(f"  r (Z lags): {self.best_lags['r']}")
        print(f"Information Criterion: {self.ic} = {self.best_ic:.4f}")
        print()
        
        print(self.model.summary())
        
        # Long-run multipliers
        print("\n" + "=" * 70)
        print("Long-Run Asymmetry Estimates (Delta Method)")
        print("=" * 70)
        
        df = self.model.df_resid
        
        for var in self.xnames:
            lr = self.long_run[var]
            se = self.long_run_se[var]
            
            t_pos = lr['positive'] / se['positive']
            t_neg = lr['negative'] / se['negative']
            
            p_pos = 2 * (1 - stats.t.cdf(abs(t_pos), df))
            p_neg = 2 * (1 - stats.t.cdf(abs(t_neg), df))
            
            sig_pos = '***' if p_pos < 0.001 else '**' if p_pos < 0.01 else '*' if p_pos < 0.05 else ''
            sig_neg = '***' if p_neg < 0.001 else '**' if p_neg < 0.01 else '*' if p_neg < 0.05 else ''
            
            print(f"\nVariable: {var}")
            print(f"  Positive LR: {lr['positive']:8.4f}  (SE: {se['positive']:6.4f}, "
                  f"t: {t_pos:6.3f}, p: {p_pos:6.4f}) {sig_pos}")
            print(f"  Negative LR: {lr['negative']:8.4f}  (SE: {se['negative']:6.4f}, "
                  f"t: {t_neg:6.3f}, p: {p_neg:6.4f}) {sig_neg}")
        
        print("\nSignif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05")
        
        # Wald tests
        print("\n" + "=" * 70)
        print("Wald Tests for Asymmetry")
        print("=" * 70)
        
        for var in self.xnames:
            print(f"\nVariable: {var}")
            if self.wald[var]['short_run'] is not None:
                print(f"  Short-run asymmetry (p-value): {self.wald[var]['short_run'].pvalue:.4f}")
            if self.wald[var]['long_run'] is not None:
                print(f"  Long-run asymmetry (p-value): {self.wald[var]['long_run'].pvalue:.4f}")
        
        # Diagnostics
        print("\n" + "=" * 70)
        print("Diagnostic Tests")
        print("=" * 70)
        
        print("\nNormality:")
        if self.diagnostics['jarque_bera']:
            jb = self.diagnostics['jarque_bera']
            print(f"  Jarque-Bera: statistic = {jb['statistic']:.4f}, "
                  f"p-value = {jb['pvalue']:.4f}")
        
        if self.diagnostics['shapiro_wilk']:
            sw = self.diagnostics['shapiro_wilk']
            print(f"  Shapiro-Wilk: statistic = {sw['statistic']:.4f}, "
                  f"p-value = {sw['pvalue']:.4f}")
        
        print("\nSerial Correlation:")
        if self.diagnostics['breusch_godfrey']:
            bg = self.diagnostics['breusch_godfrey']
            print(f"  Breusch-Godfrey: statistic = {bg['statistic']:.4f}, "
                  f"p-value = {bg['pvalue']:.4f}")
        
        print("\nHeteroskedasticity:")
        if self.diagnostics['breusch_pagan']:
            bp = self.diagnostics['breusch_pagan']
            print(f"  Breusch-Pagan: statistic = {bp['statistic']:.4f}, "
                  f"p-value = {bp['pvalue']:.4f}")
        
        if self.diagnostics['arch']:
            arch = self.diagnostics['arch']
            print(f"\nARCH Effect:")
            print(f"  ARCH(1): statistic = {arch['statistic']:.4f}, "
                  f"p-value = {arch['pvalue']:.4f}")
    
    def plot_multipliers(self, variable=None):
        """Plot dynamic multipliers"""
        if variable is None:
            variable = self.xnames[0]
        
        mult = self.dynamic_multipliers[variable]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Dynamic Multiplier - Positive
        axes[0, 0].plot(mult['positive']['horizon'], mult['positive']['multiplier'],
                        'b-', linewidth=2, label='Positive')
        axes[0, 0].axhline(0, color='r', linestyle='--', alpha=0.5)
        axes[0, 0].set_title(f'Dynamic Multiplier - {variable} (Positive)')
        axes[0, 0].set_xlabel('Horizon')
        axes[0, 0].set_ylabel('Multiplier')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Dynamic Multiplier - Negative
        axes[0, 1].plot(mult['negative']['horizon'], mult['negative']['multiplier'],
                        'darkred', linewidth=2, label='Negative')
        axes[0, 1].axhline(0, color='r', linestyle='--', alpha=0.5)
        axes[0, 1].set_title(f'Dynamic Multiplier - {variable} (Negative)')
        axes[0, 1].set_xlabel('Horizon')
        axes[0, 1].set_ylabel('Multiplier')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Cumulative Multiplier - Positive
        axes[1, 0].plot(mult['positive']['horizon'], mult['positive']['cumulative'],
                        'b-', linewidth=2, label='Cumulative')
        axes[1, 0].axhline(self.long_run[variable]['positive'], 
                          color='g', linestyle='--', linewidth=2, label='Long-run')
        axes[1, 0].set_title(f'Cumulative Multiplier - {variable} (Positive)')
        axes[1, 0].set_xlabel('Horizon')
        axes[1, 0].set_ylabel('Cumulative Effect')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Cumulative Multiplier - Negative
        axes[1, 1].plot(mult['negative']['horizon'], mult['negative']['cumulative'],
                        'darkred', linewidth=2, label='Cumulative')
        axes[1, 1].axhline(self.long_run[variable]['negative'], 
                          color='g', linestyle='--', linewidth=2, label='Long-run')
        axes[1, 1].set_title(f'Cumulative Multiplier - {variable} (Negative)')
        axes[1, 1].set_xlabel('Horizon')
        axes[1, 1].set_ylabel('Cumulative Effect')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def cumsq(e, k, n):
    """
     CUSUM of Squares Test
    
    Parameters:
    -----------
    e : array-like
        Residuals
    k : int
        Number of parameters
    n : int
        Number of observations
    """
    w = np.array(e)
    w = w[~np.isnan(w)]
    w = np.cumsum(w**2) / np.sum(w**2)
    
    m = abs(0.5 * (n - k) - 1)
    c_val = (0.74191 - 0.17459 * np.log(m) - 0.26526 * (1/m) + 
             0.0029985 * m - 0.000010943 * m**2)
    
    x = np.arange(k, k + len(w))
    w2 = c_val + (x - k) / (n - k)
    w1 = -c_val + (x - k) / (n - k)
    
    grange = [np.min([w1.min(), w2.min(), w.min()]),
              np.max([w1.max(), w2.max(), w.max()])]
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, w, color='blue', linewidth=2, label='CUSUM of squares')
    plt.plot(x, w1, color='red', linestyle='--', linewidth=1.5, label='5% significance')
    plt.plot(x, w2, color='red', linestyle='--', linewidth=1.5)
    plt.axhline(y=0.5, color='gray', linestyle=':', linewidth=1)
    plt.ylim(grange)
    plt.xlabel('Observation', fontsize=12)
    plt.ylabel('Empirical fluctuation process', fontsize=12)
    plt.title('CUSUM of Squares Test', fontsize=14, fontweight='bold')
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def cusum(e, k, n):
    """
    CUSUM Test
    
    Parameters:
    -----------
    e : array-like
        Residuals
    k : int
        Number of parameters
    n : int
        Number of observations
    """
    w = np.array(e)
    w = w[~np.isnan(w)]
    w_sd = np.std(w, ddof=1)
    w = np.cumsum(w / w_sd)
    
    c_val = 0.984
    x = np.arange(k, k + len(w))
    
    upper = c_val * np.sqrt(n - k) + (2 * c_val * np.sqrt(n - k)) * (x - k) / len(w)
    lower = -c_val * np.sqrt(n - k) + (-2 * c_val * np.sqrt(n - k)) * (x - k) / len(w)
    
    grange = [np.min([lower.min(), upper.min(), w.min()]),
              np.max([lower.max(), upper.max(), w.max()])]
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, w, color='blue', linewidth=2, label='CUSUM')
    plt.plot(x, upper, color='red', linestyle='--', linewidth=1.5, label='5% significance')
    plt.plot(x, lower, color='red', linestyle='--', linewidth=1.5)
    plt.axhline(y=0, color='gray', linestyle=':', linewidth=1)
    plt.ylim(grange)
    plt.xlabel('Observation', fontsize=12)
    plt.ylabel('Empirical fluctuation process', fontsize=12)
    plt.title('CUSUM Test', fontsize=14, fontweight='bold')
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_nardl(nardl_object, which='both'):
    """
    Plot CUSUM and/or CUSUM of Squares tests for NARDL model
    
    Parameters:
    -----------
    nardl_object : NARDL
        Fitted NARDL model object
    which : str
        Which plot to show: 'cusum', 'cusumsq', or 'both'
    """
    resid = nardl_object.model.resid.values
    n_obs = int(nardl_object.model.nobs)
    k = len(nardl_object.model.params)
    
    if which == 'both':
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # CUSUM plot
        plt.sca(ax1)
        w = resid[~np.isnan(resid)]
        w_sd = np.std(w, ddof=1)
        w = np.cumsum(w / w_sd)
        
        c_val = 0.984
        x = np.arange(k, k + len(w))
        
        upper = c_val * np.sqrt(n_obs - k) + (2 * c_val * np.sqrt(n_obs - k)) * (x - k) / len(w)
        lower = -c_val * np.sqrt(n_obs - k) + (-2 * c_val * np.sqrt(n_obs - k)) * (x - k) / len(w)
        
        grange = [np.min([lower.min(), upper.min(), w.min()]),
                  np.max([lower.max(), upper.max(), w.max()])]
        
        ax1.plot(x, w, color='blue', linewidth=2, label='CUSUM')
        ax1.plot(x, upper, color='red', linestyle='--', linewidth=1.5, label='5% significance')
        ax1.plot(x, lower, color='red', linestyle='--', linewidth=1.5)
        ax1.axhline(y=0, color='gray', linestyle=':', linewidth=1)
        ax1.set_ylim(grange)
        ax1.set_xlabel('Observation', fontsize=12)
        ax1.set_ylabel('Empirical fluctuation process', fontsize=12)
        ax1.set_title('CUSUM Test', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper left', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # CUSUM of Squares plot
        plt.sca(ax2)
        w = resid[~np.isnan(resid)]
        w = np.cumsum(w**2) / np.sum(w**2)
        
        m = abs(0.5 * (n_obs - k) - 1)
        c_val = (0.74191 - 0.17459 * np.log(m) - 0.26526 * (1/m) + 
                 0.0029985 * m - 0.000010943 * m**2)
        
        x = np.arange(k, k + len(w))
        w2 = c_val + (x - k) / (n_obs - k)
        w1 = -c_val + (x - k) / (n_obs - k)
        
        grange = [np.min([w1.min(), w2.min(), w.min()]),
                  np.max([w1.max(), w2.max(), w.max()])]
        
        ax2.plot(x, w, color='blue', linewidth=2, label='CUSUM of squares')
        ax2.plot(x, w1, color='red', linestyle='--', linewidth=1.5, label='5% significance')
        ax2.plot(x, w2, color='red', linestyle='--', linewidth=1.5)
        ax2.axhline(y=0.5, color='gray', linestyle=':', linewidth=1)
        ax2.set_ylim(grange)
        ax2.set_xlabel('Observation', fontsize=12)
        ax2.set_ylabel('Empirical fluctuation process', fontsize=12)
        ax2.set_title('CUSUM of Squares Test', fontsize=14, fontweight='bold')
        ax2.legend(loc='upper left', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    elif which == 'cusum':
        cusum(resid, k, n_obs)
    elif which == 'cusumsq':
        cumsq(resid, k, n_obs)
    else:
        raise ValueError("which must be 'cusum', 'cusumsq', or 'both'")

