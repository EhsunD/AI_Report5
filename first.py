# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('../creditcard.csv')

data_info = df.info()
summary_stats = df.describe()

transaction_distribution = df['Class'].value_counts()
fraud_percentage = (transaction_distribution[1] / transaction_distribution.sum()) * 100

plt.figure(figsize=(8, 6))
sns.countplot(x='Class', data=df)
plt.title('Distribution of Classes')
plt.show()

plt.figure(figsize=(14, 8))
sns.boxplot(data=df.drop(['Class'], axis=1))
plt.title('Boxplot of Features')
plt.xticks(rotation=90)
plt.show()
