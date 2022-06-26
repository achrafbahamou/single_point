import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook
import matplotlib as mpl
import seaborn as sns
import warnings
warnings.filterwarnings("ignore",category=mpl.cbook.mplDeprecation)
warnings.filterwarnings(action='once')

large = 22; med = 16; small = 12
params = {'axes.titlesize': large,
          'legend.fontsize': med,
          'figure.figsize': (16, 10),
          'axes.labelsize': med,
          'axes.titlesize': med,
          'xtick.labelsize': med,
          'ytick.labelsize': med,
          'figure.titlesize': large}
plt.rcParams.update(params)
plt.style.use('seaborn-whitegrid')
sns.set_style("whitegrid")

def plot_opt_mech(ai_s, sol, logs = True):
    fig, ax = plt.subplots(figsize = (6,6))
    fig.patch.set_facecolor('white')

    probas =  np.array(sol['x']).reshape(-1)[:-1]

    ax.bar(ai_s, probas, width = 0.7*np.append([ai_s[i+1]-ai_s[i] for i in range(len(ai_s)-1)], [ai_s[-1]-ai_s[-2]]), 
           align = 'edge', edgecolor = 'b', alpha = 1, color = 'b')

    ax.set_xlabel(r'$a_i$', fontsize=20)
    ax.set_ylabel(r'$d \Psi(a_i)$', fontsize=20)

    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)

    ax.xaxis.set_tick_params(top='on', direction='in', width=2)
    ax.yaxis.set_tick_params(right='on', direction='in', width=2)
    
    if logs:
        ax.set_xscale('log')
        ax.set_yscale('log')
        
    plt.show()