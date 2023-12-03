from scipy import sqrt, log, exp
import matplotlib.pyplot as plt

# radius of MoM confidence sphere
# lugosi mendelson survey prop 1

def momradius(n, alpha, trSigma):
    return 4*sqrt(trSigma*(8*log(1/alpha)+1)/n)


# radius of our Catoni-Giulini confidence sphere sequence
def CGradius(t, alpha, trSigma, sumlambda, sumlambdasq, beta=1):
    return (trSigma*(2*exp(2/beta + 2)+1)*sumlambdasq + beta/2 + log(1/alpha))/sumlambda



from matplotlib import rc

# Configure Matplotlib to use LaTeX
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
plt.rcParams.update({'font.size': 18})



alpha = 0.05

momradii = [ ]
CGradii = [ ]

sumlambda = 0
sumlambdasq = 0

d = 10
end = 10000

for t in range(1,end+1):
    sumlambda += sqrt( log(1/alpha)/(d*t*log(t+400)*10) )
    sumlambdasq += log(1/alpha)/(d*t*log(t+400)*10)
    momradii.append(momradius(t, alpha/(t+t*t), d))
    CGradii.append(CGradius(t, alpha, d, sumlambda, sumlambdasq, beta=8))

start = 150


plt.figure(figsize=(5,2))
plt.plot(range(start, end), momradii[start:], label='MoM', c='blue', lw=2, ls='--')
plt.plot(range(start, end), CGradii[start:], label='Theorem 3', c='red', lw=2, ls='-')

plt.legend(fontsize=12)
# plt.xscale('log')
plt.xlim()
plt.ylabel('CSS Radius')
plt.xlabel('Samples')
#plt.xticks(fontsize=8)
#plt.yticks(fontsize=8)
plt.savefig('cgvsmom.pdf', dpi=300, bbox_inches='tight')
