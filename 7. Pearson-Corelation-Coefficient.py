import pandas as pd
from scipy.stats import pearsonr
data={
    "Age":[10,20,30,40],
    "Marks":[5,6,3,1]
}
df = pd.DataFrame(data)
pp = pearsonr(df['Age'], df['Marks'])
print('Pearsons correlation:',pp[0])
