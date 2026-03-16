import matplotlib.pyplot as plt
import scipy as sp
from scipy.integrate import odeint
import numpy as np
from reproductive_number import *
from scipy.stats import qmc

def odes (x, t): 

    A = x[0]
    P = x[1]
    C = x[2]
    I = x[3]

    # our ODEs 
    dPdt = lambda_p - S_p * P - Beta * C * P   #good 
    dCdt = u_c * (1 - alpha_c) * C  - S_c * (1+ alpha_c) * I * C + Beta * C * P #good
    dAdt = S_p * P + S_c * (1 + alpha_c) * I * C - r * I * A #good
    dIdt = lambda_i - S_i * I + u_c * C #good 

    return [dAdt, dPdt, dCdt, dIdt]

#Sensitivity of parameters:
S_alphaC = (alpha_c / r_0)* ((S_i*(-1)*(2*S_p*u_c - Beta*lambda_p)) / (S_p*S_c*lambda_i*(1+alpha_c)**2))

S_beta = (Beta / r_0)*((lambda_p*S_i)/(S_p*S_c*(1+alpha_c)*lambda_i))

S_deltaC = (S_c / r_0)*(-1)*(S_i*(u_c*(1-alpha_c)*S_p+Beta*lambda_p)/(((S_c)**2)*(1+alpha_c)*lambda_i*S_p))

S_deltaI = (S_i / r_0)*((u_c*(1-alpha_c)*S_p+Beta*lambda_p)/(S_c*S_p*(1+alpha_c)*lambda_i))

S_deltaP = (S_p / r_0)*(-1)*((Beta*lambda_p*S_i) / ((S_c*(S_p)**2) *(1+alpha_c)*lambda_i))

S_lambdaI = (lambda_i / r_0)*(-1)*S_i*((u_c*(1-alpha_c)+Beta*(lambda_p / S_p))/(S_c*(1+alpha_c)*((lambda_i)**2)))

S_lambdaP = (lambda_p / r_0)*(Beta*S_i/S_p*S_c*(1+alpha_c)*lambda_i)

S_muC = (u_c/r_0)*((S_i*(1-alpha_c))/(S_c*(1+alpha_c)*lambda_i))

#gather all sensitivities
params = dict(aC = S_alphaC, B = S_beta, dC = S_deltaC, dI = S_deltaI, dP = S_deltaP, li = S_lambdaI, lP = S_lambdaP, mC = S_muC)


#cancer data from tsc
numCancerPercent = [86.852974,84.4924812,105.662594,12.5704887,48.9074248,66.0244361,6.93139098,12.3237782,27.2321429,14.0666667,33.4555556,65.6222222,18.8111111,28.6,25.2888889,100,6.1325188]
#scaling from percent to #cells
scaledCancer = numCancerPercent*400

#3 different kinds of means of cancer data
cancerPercentMeans = [31.38,42.4,22.0]*400

# x0=[A(0), P(0), C(0), I(0)]
x_0 = [4000, 1.75e6, 40000, 10e6] 


print(odes(x_0,0))
tf = 120
t = np.linspace(0,tf,1000)
x = odeint(odes, x_0, t)

A = x[:,0]
P = x[:,1]
C = x[:,2]
I = x[:,3]

#Running Fake Simulations
sampler = qmc.LatinHypercube(d=8)
sample = sampler.random(n=10)
#print(f'Sample before scaling is {sample}')

l_bounds = [1e-25, Beta - 0.1*Beta, S_c - 0.1*S_c, S_p - 0.1*S_p, S_i - 0.1*S_i, lambda_p - 0.1*lambda_p, lambda_i - 0.1*lambda_i, u_c - 0.1*u_c]
u_bounds = [2e-25, Beta + 0.1*Beta, S_c + 0.1*S_c, S_p + 0.1*S_p, S_i + 0.1*S_i, lambda_p + 0.1*lambda_p, lambda_i + 0.1*lambda_i, u_c + 0.1*u_c]
scaled_sample = qmc.scale(sample, l_bounds, u_bounds)
print(f'Scaled LHS sample is {scaled_sample}')

# # Plotting each equation in a separate subplot with a log y-axis
fig, axis = plt.subplots(3, 2, figsize=(7, 7))

# Plot P(t)
axis[0, 0].plot(t, P, color='blue')
axis[0, 0].set_title("$P(t)$ Graph")
axis[0, 0].set_yscale('log')
axis[0, 0].set_xlabel('$t$')
axis[0, 0].set_ylabel('$P(t)$')
axis[0, 0].grid()

# Plot A(t)
axis[0, 1].plot(t, A, color='green')
axis[0, 1].set_title("$A(t)$ Graph")
axis[0, 1].set_yscale('log')
axis[0, 1].set_xlabel('$t$')
axis[0, 1].set_ylabel('$A(t)$')
axis[0, 1].grid()

# Plot C(t)
axis[1, 0].plot(t, C, color='red')
axis[1, 0].set_title("$C(t)$ Graph")
axis[1, 0].set_yscale('log')
axis[1, 0].set_xlabel('$t$')
axis[1, 0].set_ylabel('$C(t)$')
axis[1, 0].grid()

#Ideal Data Points
x_coords = np.array ([0, 15, 30, 45, 60, 75, 90, 105, 120])
y_coords = np.array([40000, 44000, 48000, 52500, 50000, 54500, 56750, 58000, 60000])
ideal_data= (x_coords,y_coords)
axis[1,0].plot(x_coords, y_coords, 'o')

#Hypercube Sampling Data 
#axis[1,0].plot(scaled_sample, 'o')
#for d in range (10):
    #differences= np.abs(scaled_sample - ideal_data)
    #sum_of_differences= np.sum(differences)
    
    
#plot mtt data
for i in scaledCancer:
    axis[1,0].plot(120,i,'xk')
#plot means
for j in cancerPercentMeans:
    axis[1,0].plot(120,j,".c")

# Plot I(t)
axis[1,1].plot(t, I, color='purple')
axis[1,1].set_title('$I(t)$ Graph')
axis[1,1].set_yscale('log')
axis[1,1].set_xlabel('$t$')
axis[1,1].set_ylabel('$I(t)$')
axis[1,1].grid()

#plot sensitivity
x = params.keys()
y = params.values()
axis[2,0].bar(x, y, width=.5)

#make last spot empty
axis[2,1].axis("off")

# Adjust layout for better spacing
plt.tight_layout()
plt.show()