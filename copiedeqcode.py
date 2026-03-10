import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import scipy as sp
from scipy.integrate import odeint
import numpy as np
from scipy.stats import qmc

'''
    IDEAL DATA POINTS
'''
x_coords = np.array([0, 15, 30, 45, 60, 75, 90, 105, 120])
y_coords = np.array([40000, 44000, 48000, 52500, 50000, 54500, 56750, 58000, 60000])

'''
    COEFFICIENT VALUES
'''
lambda_p = 5
lambda_c = 10
lambda_i = .5
Beta = .00125
alpha_c = .5
alpha_i = .2
S_pc = .9
S_pn = 9e-6
S_i = 2
r = 1.8
n = .8

'''
    MODEL FUNCTIONS
    Passes state vector x of type list 
    Passes time t of type float
'''
def odes(x: list, t: float) -> list: 

    P_n = x[0]
    P_c = x[1]
    I   = x[2]
    D_n = x[3]
    D_c = x[4]
    
    '''
        ODE's for proliferative, dead, and immune cells
    '''
    dP_ndt = lambda_p - Beta * I * P_n - S_pn * P_n
    dP_cdt = lambda_c * P_c + Beta * I * P_n - alpha_i * I * P_c - S_pc * P_c
    dIdt   = lambda_i + alpha_c * P_c - S_i * I
    dD_ndt = S_pn * P_n - r * D_n
    dD_cdt = S_pc * P_c + alpha_i * I * P_c - n * r*I * D_c
    

    return [dP_ndt, dP_cdt, dIdt, dD_ndt, dD_cdt]

'''
    DEFINE PARAMTER VALUES
'''

# x0=[P_n, P_c, I, D_n, D_c]
x_0 = [1.75e5, 40, 10, 0, 0] 

#odeint

t = np.linspace(0,120,1000)
x = odeint(odes, x_0, t)

P_n = x[:,0]
P_c = x[:,1]
I   = x[:,2]
D_n = x[:,3]
D_c = x[:,4]

#Define Cancer
C = P_c + D_c

#Parameter vector
params = np.array([lambda_p, lambda_c, lambda_i, Beta, alpha_c, alpha_i, S_pc, S_pn, S_i, r, n, x_0[0], x_0[1], x_0[2]])

#Latin Hypercube
sampler = qmc.LatinHypercube(d=(len(params)))
sample = sampler.random(n=100)
#print(f'Sample before scaling is {sample}')
uni_scaler = 20

# new goal, add initial conditions into our LHS stuff.  
l_bounds = [lambda_p - uni_scaler*lambda_p, lambda_c - uni_scaler*lambda_c, lambda_i - uni_scaler*lambda_i, Beta - uni_scaler*Beta, alpha_c - uni_scaler*alpha_c, alpha_i - uni_scaler*alpha_i, S_pc - uni_scaler*S_pc, S_pn - uni_scaler*S_pn,  S_i - uni_scaler* S_i, r - uni_scaler*r, n - uni_scaler*n, x_0[0] - uni_scaler*x_0[0], x_0[1] - uni_scaler*x_0[1], x_0[2] - uni_scaler*x_0[2]]
u_bounds = [lambda_p + uni_scaler*lambda_p, lambda_c + uni_scaler*lambda_c, lambda_i + uni_scaler*lambda_i, Beta + uni_scaler*Beta, alpha_c + uni_scaler*alpha_c, alpha_i + uni_scaler*alpha_i, S_pc + uni_scaler*S_pc, S_pn + uni_scaler*S_pn,  S_i + uni_scaler* S_i, r + uni_scaler*r, n + uni_scaler*n, x_0[0] + uni_scaler*x_0[0], x_0[1] + uni_scaler*x_0[1], x_0[2] + uni_scaler*x_0[2]]
scaled_sample = qmc.scale(sample, l_bounds, u_bounds)
#print(f'Scaled LHS sample is {scaled_sample}')

SSEs = np.zeros(len(scaled_sample))

for i in range(0,len(scaled_sample)):
    lambda_p   = scaled_sample[i,0] 
    lambda_c   = scaled_sample[i,1]
    lambda_i   = scaled_sample[i,2]
    Beta       = scaled_sample[i,3]
    alpha_c    = scaled_sample[i,4]
    alpha_i    = scaled_sample[i,5]
    S_pc       = scaled_sample[i,6]
    S_pn       = scaled_sample[i,7]
    S_i        = scaled_sample[i,8]
    r          = scaled_sample[i,9]
    n          = scaled_sample[i,10]
    x_0        = [scaled_sample[i,11],scaled_sample[i,12],scaled_sample[i,13], 0, 0]
    full_sim = odeint(odes,x_0,t)
    C = full_sim[:,1]+full_sim[:,4] # sum of all cancer cells
    C_sim = np.interp(x_coords, t, C)
    ERRS = C_sim - np.array(y_coords)
    SSEs[i] = np.sum(ERRS**2)

#Find the minimum SSE
best_SSE = min(SSEs[0:i])

#Find the index of the minimum SSE
best_idx = np.argmin(SSEs)

#Exact parameter values for that index
best_lambda_p   = scaled_sample[best_idx, 0]
best_lambda_c   = scaled_sample[best_idx, 1]
best_lambda_i   = scaled_sample[best_idx, 2]
best_Beta       = scaled_sample[best_idx, 3]
best_alpha_c    = scaled_sample[best_idx, 4]
best_alpha_i    = scaled_sample[best_idx, 5]
best_S_pc       = scaled_sample[best_idx, 6]
best_S_pn       = scaled_sample[best_idx, 7]
best_S_i        = scaled_sample[best_idx, 8]
best_r          = scaled_sample[best_idx, 9]
best_n          = scaled_sample[best_idx, 10]

print(f"The smallest SSE is {best_SSE}")
print(f"Best parameters at index {best_idx}:")
print(f"lambda_p   = {best_lambda_p}")
print(f"lambda_c   = {best_lambda_c}")
print(f"lambda_i   = {best_lambda_i}")
print(f"Beta       = {best_Beta}")
print(f"alpha_c    = {best_alpha_c}")
print(f"alpha_i    = {best_alpha_i}")
print(f"S_pc       = {best_S_pc}")
print(f"S_pn       = {best_S_pn}")
print(f"S_i        = {best_S_i}")
print(f"r          = {best_r}")
print(f"n          = {best_n}")

fig_plot, axs_plot = plt.subplots( 3, 2, figsize = ( 7, 7 ) )
fig_sliders = plt.figure( figsize = ( 10, 5 ) )

# Container to hold strong references to sliders so they are not garbage collected
fig_sliders.slider_refs = {}
# additional module-level list for strong refs (helps avoid GC across backends)
slider_refs_list = []

def make_slider(fig, rect, label, valinit, valmin, valmax, refs, name=None):
    ax = fig.add_axes(rect)
    s = Slider(ax=ax, label=label, valmin=valmin, valmax=valmax, valinit=valinit)
    # store strong reference
    if name:
        refs[name] = s
    else:
        refs.setdefault('list', []).append(s)
    # also keep module-level strong reference
    slider_refs_list.append(s)
    return s

'''
    PLOTS FOR 5 PARAMETERS AND TOTAL CANCER CELLS
'''

plots = [
    (axs_plot[0, 0], P_n, 'cyan', '$P_n(t)$', 'Proliferative Normal Cells'),
    (axs_plot[0, 1], P_c, 'olive', '$P_c(t)$', 'Proliferative Cancer Cells'),
    (axs_plot[1, 0], D_n, 'pink', '$D_n(t)$', 'Dead Normal Cells'),
    (axs_plot[1, 1], D_c, 'blue', '$D_c(t)$', 'Dead Cancer Cells'),
    (axs_plot[2, 0], I,   'purple', '$I(t)$', 'Immune Cells'),
    (axs_plot[2, 1], C,   'red',  '$C(t)$', 'Total Cancer Cells'),
]

'''
    SLIDERS FOR COEFFICIENTS
'''
coefficients = [lambda_p, lambda_c, lambda_i, Beta, alpha_c, alpha_i, S_pc, S_pn, S_i, r, n]

for ax, data, color, title, ylabel in plots:
    ax.plot(t, data, color = color)
    ax.set_title(title)
    ax.set_yscale('log')
    ax.set_xlabel('$t$')
    ax.set_ylabel(ylabel)
    ax.grid()

'''
    SLIDERS FOR COEFFICIENT ADJUSTMENT
'''
beta_slider = make_slider(fig_sliders, [0.06, 0.9, 0.35, 0.03], 'Beta', best_Beta, best_Beta - best_Beta * uni_scaler, best_Beta + best_Beta * uni_scaler, fig_sliders.slider_refs, name='beta_slider')
lambda_p_slider = make_slider(fig_sliders, [0.56, 0.9, 0.35, 0.03], 'Lambda_p', best_lambda_p, best_lambda_p - best_lambda_p * uni_scaler, best_lambda_p + best_lambda_p * uni_scaler, fig_sliders.slider_refs, name='lambda_p_slider')
lambda_c_slider = make_slider(fig_sliders, [0.06, 0.85, 0.35, 0.03], 'Lambda_c', best_lambda_c, best_lambda_c - best_lambda_c * uni_scaler, best_lambda_c + best_lambda_c * uni_scaler, fig_sliders.slider_refs, name='lambda_c_slider')
lambda_i_slider = make_slider(fig_sliders, [0.56, 0.85, 0.35, 0.03], 'Lambda_i', best_lambda_i, best_lambda_i - best_lambda_i * uni_scaler, best_lambda_i + best_lambda_i * uni_scaler, fig_sliders.slider_refs, name='lambda_i_slider')
alpha_c_slider = make_slider(fig_sliders, [0.06, 0.8, 0.35, 0.03], 'Alpha_c', best_alpha_c, best_alpha_c - best_alpha_c * uni_scaler, best_alpha_c + best_alpha_c * uni_scaler, fig_sliders.slider_refs, name='alpha_c_slider')
alpha_i_slider = make_slider(fig_sliders, [0.56, 0.8, 0.35, 0.03], 'Alpha_i', best_alpha_i, best_alpha_i - best_alpha_i * uni_scaler, best_alpha_i + best_alpha_i * uni_scaler, fig_sliders.slider_refs, name='alpha_i_slider')
S_pc_slider = make_slider(fig_sliders, [0.06, 0.75, 0.35, 0.03], 'S_pc', best_S_pc, best_S_pc - best_S_pc * uni_scaler, best_S_pc + best_S_pc * uni_scaler, fig_sliders.slider_refs, name='S_pc_slider')
S_pn_slider = make_slider(fig_sliders, [0.56, 0.75, 0.35, 0.03], 'S_pn', best_S_pn, best_S_pn - best_S_pn * uni_scaler, best_S_pn + best_S_pn * uni_scaler, fig_sliders.slider_refs, name='S_pn_slider')
S_i_slider = make_slider(fig_sliders, [0.06, 0.7, 0.35, 0.03], 'S_i', best_S_i, best_S_i - best_S_i * uni_scaler, best_S_i + best_S_i * uni_scaler, fig_sliders.slider_refs, name='S_i_slider')
r_slider = make_slider(fig_sliders, [0.56, 0.7, 0.35, 0.03], 'r', best_r, best_r - best_r * uni_scaler, best_r + best_r * uni_scaler, fig_sliders.slider_refs, name='r_slider')
n_slider = make_slider(fig_sliders, [0.06, 0.65, 0.35, 0.03], 'n', best_n, best_n - best_n * uni_scaler, best_n + best_n * uni_scaler, fig_sliders.slider_refs, name='n_slider')

sliders = [fig_sliders.slider_refs[k] for k in ['beta_slider','lambda_p_slider','lambda_c_slider','lambda_i_slider','alpha_c_slider','alpha_i_slider','S_pc_slider','S_pn_slider','S_i_slider','r_slider','n_slider']]

def update(val):
    current_beta = beta_slider.val
    current_lambda_p = lambda_p_slider.val
    current_lambda_c = lambda_c_slider.val
    current_lambda_i = lambda_i_slider.val
    current_alpha_c = alpha_c_slider.val
    current_alpha_i = alpha_i_slider.val
    current_S_pc = S_pc_slider.val
    current_S_pn = S_pn_slider.val
    current_S_i = S_i_slider.val
    current_r = r_slider.val
    current_n = n_slider.val

    dP_ndt = current_lambda_p - current_beta * I * P_n - current_S_pn * P_n
    dP_cdt = current_lambda_c * P_c + current_beta * I * P_n - current_alpha_i * I * P_c - current_S_pc * P_c
    dIdt   = current_lambda_i + current_alpha_c * P_c - current_S_i * I
    dD_ndt = current_S_pn * P_n - current_r * D_n
    dD_cdt = current_S_pc * P_c + current_alpha_i * I * P_c - current_n * current_r*I * D_c
    
    axs_plot[0, 0].lines[0].set_ydata(dP_ndt)
    axs_plot[0, 1].lines[0].set_ydata(dP_cdt)
    axs_plot[1, 0].lines[0].set_ydata(dD_ndt)
    axs_plot[1, 1].lines[0].set_ydata(dD_cdt)
    axs_plot[2, 0].lines[0].set_ydata(dIdt)

    fig_plot.canvas.draw_idle()
    # also redraw slider figure to keep widgets responsive across backends
    try:
        fig_sliders.canvas.draw_idle()
    except Exception:
        pass

for s in sliders:
    s.on_changed(update)

reset_axs = fig_sliders.add_axes([0.8, 0.05, 0.1, 0.04])
button = matplotlib.widgets.Button(reset_axs, 'Reset', color='lightgray', hovercolor='gray')

def reset(event):
    for s in sliders:
        try:
            s.reset()
        except Exception:
            pass
    
button.on_clicked(reset)

plt.show()