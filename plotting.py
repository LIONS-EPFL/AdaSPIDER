import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ada_spider = pd.read_csv("losses-AdaSpider(step_size=0.01, n=50000).csv")
sgd = pd.read_csv("losses-SGD(step_size=0.01).csv")


plt.style.use('bmh')

plt.plot(ada_spider, label="spider", ls='-.')
plt.plot(sgd, label='sgd')
plt.legend()
plt.yscale('log')
plt.ylabel("loss")
plt.show()