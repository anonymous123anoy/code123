import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.ndimage import gaussian_filter1d




def plot_mean_and_conf(ax, data, labels, conf=0.95, smooth_window=10,
                      ylabel='', xlabel='', title='', fontsize=14, 
                      alpha=0.4, smooth_method='gaussian', smooth_ci=True, linewidth=3.0):

    for i, experiment in enumerate(data):
        means = np.mean(experiment, axis=0)
        sems = stats.sem(experiment, axis=0)
        ci = sems * stats.t.ppf((1 + conf) / 2, experiment.shape[0] - 1)
        
        timesteps = np.arange(1, experiment.shape[1] + 1)
        
        # Smoothing
        if smooth_method == 'gaussian':
            means_smoothed = gaussian_filter1d(means, sigma=smooth_window/3)
            ci_smoothed = gaussian_filter1d(ci, sigma=smooth_window/3) if smooth_ci else ci
        elif smooth_method == 'mean':
            means_smoothed = moving_average(means, smooth_window)
            ci_smoothed = moving_average(ci, smooth_window) if smooth_ci else ci
        else:
            raise ValueError("smooth_method must be 'gaussian' or 'mean'")
        
        # Plot
        ax.plot(timesteps, means_smoothed, label=labels[i] if labels else None, linewidth=linewidth)
        ax.fill_between(timesteps, means_smoothed + ci_smoothed, 
                        means_smoothed - ci_smoothed, alpha=alpha)
    
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize + 2)
    ax.grid(True)

def moving_average(vec, window):
    pad = window // 2
    vec_padded = np.pad(vec, (pad, pad), mode='edge')
    return np.convolve(vec_padded, np.ones(window)/window, mode='valid')



dataset = "synthetic"
metrics = ['entropy', 'proportion', 'neighbor', 'disvar', 'maxsim']
metrics_name = ["RCE", "RA", "ND", "PDV", "TS@50"]


hyper_name = "c"

if hyper_name == "alpha":
    hyper_list = [0.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.5, 10.0]
    data_list = ["eva_dict_seeds_alpha_0.0.npy", "eva_dict_seeds_alpha_2.5.npy", "eva_dict_seeds_alpha_3.0.npy", "eva_dict_seeds_alpha_3.5.npy", "eva_dict_seeds_alpha_4.0.npy", "eva_dict_seeds_alpha_4.5.npy", "eva_dict_seeds_alpha_5.0.npy", "eva_dict_seeds_alpha_6.0.npy", "eva_dict_seeds_alpha_7.5.npy", "eva_dict_seeds_alpha_10.0.npy"]

elif hyper_name == "beta":
    hyper_list = [0.0, 0.1, 0.5, 1.5, 2.5, 3.5, 4.5, 10.0]
    data_list = ["eva_dict_seeds_beta_0.0.npy", "eva_dict_seeds_beta_0.1.npy", "eva_dict_seeds_beta_0.5.npy", "eva_dict_seeds_beta_1.5.npy", "eva_dict_seeds_beta_2.5.npy", "eva_dict_seeds_beta_3.5.npy", "eva_dict_seeds_beta_4.5.npy", "eva_dict_seeds_beta_10.0.npy"]

elif hyper_name == "gamma":
    hyper_list = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    data_list = ["eva_dict_seeds_gamma_0.0.npy", "eva_dict_seeds_gamma_0.2.npy", "eva_dict_seeds_gamma_0.4.npy", "eva_dict_seeds_gamma_0.6.npy", "eva_dict_seeds_gamma_0.8.npy", "eva_dict_seeds_gamma_1.0.npy"]

elif hyper_name == "epsilon":
    hyper_list = [-0.15, -0.1, -0.06, -0.03, 0.0, 0.03, 0.06, 0.1, 0.15]
    data_list = ["eva_dict_seeds_epsilon_n0.15.npy", "eva_dict_seeds_epsilon_n0.1.npy", "eva_dict_seeds_epsilon_n0.06.npy", "eva_dict_seeds_epsilon_n0.03.npy", "eva_dict_seeds_alpha_5.0.npy", "eva_dict_seeds_epsilon_0.03.npy", "eva_dict_seeds_epsilon_0.06.npy", "eva_dict_seeds_epsilon_0.1.npy", "eva_dict_seeds_epsilon_0.15.npy"]

elif hyper_name == "s":
    hyper_list = [500, 1000, 2500, 5000, 10000, 20000, 40000, 100000, 200000]
    data_list = ["eva_dict_seeds_social_500.npy", "eva_dict_seeds_social_1000.npy", "eva_dict_seeds_social_2500.npy", "eva_dict_seeds_social_5000.npy", "eva_dict_seeds_alpha_5.0.npy", "eva_dict_seeds_social_20000.npy", "eva_dict_seeds_social_40000.npy", "eva_dict_seeds_social_100000.npy", "eva_dict_seeds_social_200000.npy"]

elif hyper_name == "m":
    hyper_list = [500, 750, 1000, 2500, 5000, 10000, 100000]
    data_list = ["eva_dict_seeds_item_500.npy", "eva_dict_seeds_item_750.npy", "eva_dict_seeds_item_1000.npy", "eva_dict_seeds_item_2500.npy", "eva_dict_seeds_item_5000.npy", "eva_dict_seeds_alpha_5.0.npy", "eva_dict_seeds_item_100000.npy"]

elif hyper_name == "c":
    hyper_list = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    data_list = ["eva_dict_seeds_cate_5.npy", "eva_dict_seeds_cate_6.npy", "eva_dict_seeds_cate_7.npy", "eva_dict_seeds_cate_8.npy", "eva_dict_seeds_cate_9.npy", "eva_dict_seeds_alpha_5.0.npy", "eva_dict_seeds_cate_11.npy", "eva_dict_seeds_cate_12.npy", "eva_dict_seeds_cate_13.npy", "eva_dict_seeds_cate_14.npy"]


def format_sci(k):
    if k < 100:
        return f"{k}"
    exp = int(np.log10(k))
    base = k / (10 ** exp)
    if base.is_integer():
        return rf"{int(base)} \times 10^{{{exp}}}"
    else:
        return rf"{base:.1f} \times 10^{{{exp}}}".replace(".0 ", " ")  


if hyper_name == "s":
    labels = [r'$|\mathbf{S}|=$' + rf'${format_sci(k)}$' for k in hyper_list]
elif hyper_name == "m" or hyper_name == "c":
    labels = [rf'${hyper_name}={format_sci(k)}$' for k in hyper_list]
else:
    labels = [rf'$\{hyper_name}={k}$' for k in hyper_list]
fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 7))



for i, metric in enumerate(metrics):
    metric_name = metrics_name[i]
    metric_array_list = []
    for data in data_list:
        data_dict = np.load(dataset + "/" + data, allow_pickle=True).item()
        arr = np.expand_dims(np.array([data_dict[j][metrics[i]] for j in data_dict.keys()]), axis=0)
        metric_array_list.append(arr)
    metric_array = np.concatenate(metric_array_list, axis=0)

    plot_mean_and_conf(
        axes[i], metric_array, labels, 
        xlabel=r"Time $t$",  
        title=metric_name,
        smooth_window=30,
        fontsize=21, 
        linewidth=4.0
    )
    axes[i].tick_params(axis='both', which='major', labelsize=18)

handles, labels = axes[0].get_legend_handles_labels()
if labels:  
    if hyper_name == "alpha":
        leg = fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.09),
                ncol=len(labels), fontsize=20)
    elif hyper_name == "epsilon":
        leg = fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.09),
                ncol=len(labels), fontsize=21)
    elif hyper_name == "s":
        leg = fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.09),
                ncol=len(labels), fontsize=18)
    else:
        leg = fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.1),
                ncol=len(labels), fontsize=23)
    for line in leg.get_lines():
        line.set_linewidth(4.0)  
plt.tight_layout()
plt.savefig(f"{dataset}/{dataset}_{hyper_name}_metrics.pdf", bbox_inches='tight', dpi=300)
plt.show()