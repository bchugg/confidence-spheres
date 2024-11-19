import matplotlib.pyplot as plt
from matplotlib import rc
import math 
import numpy as np 

log = np.log 
exp = np.exp 
sqrt = np.sqrt

# radius of MoM confidence sphere
# lugosi mendelson survey prop 1

def momradius(n, alpha, trSigma):
    return 4*sqrt(trSigma*(8*log(1/alpha)+1)/n)


# radius of our Catoni-Giulini confidence sphere sequence
def CGradius(t, alpha, trSigma, sumlambda, sumlambdasq, beta=1):
    return (np.sqrt(trSigma)*(2*exp(2/beta + 2)+1)*sumlambdasq + beta/2 + log(1/alpha))/sumlambda


def Duchi_Haque(t, alpha, trSigma):
    # compute n = log base 2 of t 
    #n = max(math.log(t, 2), 1)

    return 4*np.sqrt(trSigma*(8*(log(1/alpha) + 2*log(max(math.log(t,2),1)) + 1/2) +1)/(t/2))



# Configure Matplotlib to use LaTeX
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
plt.rcParams.update({'font.size': 18})



alpha = 0.05

momradii = [ ]
CGradii = [ ]
DHradii = []

sumlambda = sumlambdasq = 0

d = 20
end = 10000

for t in range(1,end+1):
    sumlambda += sqrt( log(1/alpha)/(sqrt(d)*t*log(t+10)*4) )
    sumlambdasq += log(1/alpha)/(sqrt(d)*t*log(t+10)*4)
    momradii.append(momradius(t, alpha/(t+t*t), d))
    CGradii.append(CGradius(t, alpha, d, sumlambda, sumlambdasq, beta=4))
    DHradii.append(Duchi_Haque(t, alpha, d))

start = 100

#print(DHradii)


# plt.figure(figsize=(5,2))
plt.plot(range(start, end), momradii[start:], label='MoM', c='blue', lw=2, ls='--')
plt.plot(range(start, end), CGradii[start:], label='Theorem 6', c='red', lw=2, ls='-')
plt.plot(range(start, end), DHradii[start:], label='DH', c='purple', lw=2, ls='dotted')

plt.legend(fontsize=16)
# plt.xscale('log')
plt.xlim()
plt.ylabel('CSS Radius')
plt.xlabel('Samples')
#plt.xticks(fontsize=8)
plt.xscale('log')
#plt.yticks(fontsize=8)
#plt.savefig('cgvsmom.pdf', dpi=300, bbox_inches='tight')
plt.savefig('cgvsmom_dh.png', dpi=300, bbox_inches='tight')
