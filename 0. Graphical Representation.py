import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("flipkart-product-data.csv")
print(df)
plt.scatter(df['Price'],df['Off(%)'])
plt.show()