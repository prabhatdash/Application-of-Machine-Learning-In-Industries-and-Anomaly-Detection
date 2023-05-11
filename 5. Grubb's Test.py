import numpy as np
from outliers import smirnov_grubbs as grubbs
import matplotlib.pyplot as plt
# !pip install outlier_utils
# define data
data = np.array([20, 21, 26, 24, 29,50, 22, 21, 28, 27])
clean_data=grubbs.test(data,alpha=0.5)
plt.plot(data)
plt.plot(clean_data)
plt.show()
print(data)
print(clean_data)
