import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.linalg import norm
from scipy.special import iv
from scipy.stats import beta


from matplotlib import rc

# Configure Matplotlib to use LaTeX
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
plt.rcParams.update({'font.size': 18})

def gen_unif_vec(d):
    return beta.rvs(1, 1, size=d)

def gen_beta1_vec(d):
    return beta.rvs(10, 10, size=d)

def gen_beta2_vec(d):
    return beta.rvs(50, 10, size=d)


def lambd(i, alpha, sigma2):

    top = 8 * np.log(1/alpha)
    bottom = i * np.log(i+1) * sigma2
    return min(np.sqrt(top / bottom), 0.5)

def psiE(x):
    assert 0 < x < 1
    return -x - np.log(1 - x)

def A(d, kappa):
    return iv(d/2, kappa) / iv(d/2-1, kappa)


def emp_bernstein(alpha, B, d, iters, gen_vec):

    adk = A(d, np.sqrt(d))

    emp_mean_numerator = 1/2
    running_numerator = 4*B*np.sqrt(d)*adk + 2*B*np.sqrt(d)*np.log(1/alpha)
    running_denominator = 0

    sigma_hat_top = 1/4

    width = []

    for i in range(1,int(iters)):

        # Compute lambda
        lmd = lambd(i, alpha, sigma_hat_top/i)

        # Get observations
        newX = gen_vec(d)

        # Update width
        running_numerator += psiE(lmd) * (norm(newX - emp_mean_numerator / i))**2
        running_denominator += adk*lmd

        width.append(min(running_numerator/running_denominator, 1))

        # Update sigma and empirical mean
        emp_mean_numerator += newX
        sigma_hat_top += norm(newX - (emp_mean_numerator / (i+1)))**2

    return width


def subG_boundary(d, alpha, sigma, iters):

    num = np.sqrt(d)*2 + np.sqrt(d) * np.log(1/alpha)
    denom = 0

    width = []
    for i in range(1, int(iters)):

        lmb = np.sqrt(np.log(1/alpha) / (i * np.log(i+1) * sigma**2))
        num += np.sqrt(d) * (sigma**2) * (lmb**2) / 2
        denom += (2/3) * lmb
        width.append(min(num /denom, 1))

    return width


alpha = 0.05
d = 5
B = d/2
iters = 10e3

width_beta1 = emp_bernstein(alpha, B, d, iters, gen_beta1_vec)
width_beta2 = emp_bernstein(alpha, B, d, iters, gen_beta2_vec)
width_unif = emp_bernstein(alpha, B, d, iters, gen_unif_vec)


start = 150


plt.figure(figsize=(5,2))

plt.plot(range(start, int(iters)), width_beta1[start-1:], label='Beta(10,10)', c='purple', lw=2, ls='-')
plt.plot(range(start, int(iters)), width_beta2[start-1:], label='Beta(50,10)', c='tab:olive', lw=2, ls='-')
plt.plot(range(start, int(iters)), width_unif[start-1:], label='Beta(1,1)', c='green', lw=2, ls='-')
plt.legend(fontsize=12)
# plt.xscale('log')
plt.xlim()
plt.ylabel('CSS Radius')
plt.xlabel('Samples')
#plt.xticks(fontsize=14)
#plt.yticks(fontsize=14)
plt.savefig('emp_bern_bounds_beta.pdf', dpi=300, bbox_inches='tight')
