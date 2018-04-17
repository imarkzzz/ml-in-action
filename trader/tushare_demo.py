import matplotlib.pylab as plt
import seaborn as sns
import seaborn.linearmodels as snsl
import tushare as ts
sns.set_style("whitegrid")
stock = ts.get_hist_data('601375', '2017-01-01', '2017-12-01')
stock['close'].plot(legend=True, figsize=(10, 4))
stock['open'].plot(legend=True, figsize=(10, 4))
plt.show()
