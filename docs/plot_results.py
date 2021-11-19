import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('docs/results.csv',index_col='DATASET')
df.plot.bar()
plt.xticks(rotation='horizontal')
plt.show()