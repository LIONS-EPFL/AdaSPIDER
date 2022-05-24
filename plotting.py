# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: 'Python 3.8.3 (''virt'': venv)'
#     language: python
#     name: python3
# ---

# %%
import wandb
api = wandb.Api()


runs = api.runs("epfl-lions/Neurips-AdaSpider")
summary_list = [] 
config_list = [] 
name_list = [] 
for run in runs: 
    config = {k:v for k,v in run.config.items()}
    for row in run.scan_history():
        if 'loss' in row:
            continue
        summary_list.append(row)
        config_list.append(config)
        name_list.append(run.name) 

import pandas as pd 
summary_df = pd.DataFrame.from_records(summary_list) 
config_df = pd.DataFrame.from_records(config_list) 
name_df = pd.DataFrame({'name': name_list}) 
all_df = pd.concat([name_df, config_df,summary_df], axis=1)

all_df.to_csv("project.csv")

# %%
means = all_df[all_df["dataset"]=="MNIST"].groupby(["optimizer", "epoch"])["grad_norm"].mean()
stds = all_df[all_df["dataset"]=="MNIST"].groupby(["optimizer", "epoch"])["grad_norm"].std()

# %%
import matplotlib.pyplot as plt
import numpy as np

plt.style.use("seaborn-poster")
plt.style.use('bmh')
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.size'] = '16'
styles = ['solid', 'dotted', 'dashed', 'dashdot']
markers = ['.', 'v', '2', '8', 'P', '*', 'X', 's']
optimizers = ["AdaGrad", "SGD", "KatyushaXw", "AdaSVRG", "Spider", "AdaSpider", "SpiderBoost"]

for i, opt in enumerate(optimizers):
    plt.plot(468*means[opt].index, means[opt], label=opt, ls=styles[i%len(styles)], marker=markers[i])
    plt.fill_between(468*means[opt].index,means[opt] - stds[opt]/np.sqrt(5) , means[opt] + stds[opt]/np.sqrt(5), alpha=0.1)


plt.yscale('log')
plt.ylabel(r'$\| \nabla f (x_k)\|^2$')
plt.xlabel("# stochastic oracle calls")
plt.legend()
plt.savefig("MNIST-grad-norm.pdf", format='pdf')
plt.show()
    

# %%
# %pwd

# %%
