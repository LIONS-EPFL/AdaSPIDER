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

# Project is specified by <entity/project-name>
runs = api.runs("epfl-lions/Neurips-AdaSpider")
summary_list = [] 
config_list = [] 
name_list = [] 
for run in runs: 
    
    # run.summary are the output key/values like accuracy.
    # We call ._json_dict to omit large files  
    
    # run.config is the input metrics.
    # We remove special values that start with _.
    config = {k:v for k,v in run.config.items()}
    

    # run.name is the name of the run.
      
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
means = all_df[all_df["dataset"]=="FashionMNIST"].groupby(["optimizer", "epoch"])["grad_norm"].mean()
stds = all_df[all_df["dataset"]=="FashionMNIST"].groupby(["optimizer", "epoch"])["grad_norm"].std()

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
    plt.plot(means[opt].index, means[opt], label=opt, ls=styles[i%len(styles)], marker=markers[i])
    plt.fill_between(means[opt].index,means[opt] - stds[opt]/np.sqrt(5) , means[opt] + stds[opt]/np.sqrt(5), alpha=0.1)


plt.yscale('log')
plt.ylabel(r'$\| \nabla f (x_k)\|$')
plt.xlabel("epoch")
#plt.title("Gradient norms")
plt.legend()
plt.savefig("FashionMNISTGradNorm.pdf", format='pdf')
plt.show()
    
