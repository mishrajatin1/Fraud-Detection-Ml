import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



df= pd.read_csv("Fraud.csv")
#print(df.info())

plt.figure(figsize=(8,5))
sns.set_theme(style="whitegrid")

sns.countplot(x='type', data=df, palette='mako', edgecolor='black')

plt.title("Types of Transactions", fontsize=16, fontweight='bold', color='navy')
plt.ylabel("Number of Transactions", fontsize=12)
plt.xticks(rotation=15)
plt.ticklabel_format(style='plain', axis='y')

plt.grid(axis='y', linestyle='--', alpha=0.6)

#Add values on top of bars
ax = plt.gca()
for p in ax.patches:
     ax.annotate(f'{int(p.get_height()):,}', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()

frauds_vs_non = df['isFraud'].value_counts()
#print(frauds_vs_non)
plt.figure(figsize=(8,6))
sns.set_theme(style="whitegrid")
plt.pie(frauds_vs_non.values ,labels=['Non Fraud' , 'Fraud'] , autopct= '%1.3f%%',startangle=90,colors=('Green','White'),textprops={'fontsize': 11})
plt.title("Fraud VS NON fraud Transaction Distribution")
plt.legend(labels=['Non Fraud ' , 'Fraud'],loc='upper right')
plt.show()
#print(df.head(10))
#print(df.shape)



fraud_counts = df['isFraud'].value_counts()
print(fraud_counts)






fraud_by_type = df.groupby('type')['isFraud'].mean().sort_values(ascending=False)
fraud_by_type.plot(kind= 'bar' ,title="Fraud Transaction By Amount Type ",color='salmon')
plt.ylabel("Fraud rate")
plt.show()


sns.boxplot(data=df[df['amount']<50000],x='isFraud' , y ='amount')
plt.title("Amount Distribution by Fraud Status")
plt.show()


fraud_per_step = df[df['isFraud']==1]['step'].value_counts().sort_index()
#print(fraud_per_step)

plt.plot(fraud_per_step.index , fraud_per_step.values )
plt.title("Frauds Per Step")
plt.xlabel("Steps[Time]")
plt.ylabel("Number of Frauds")
plt.grid(True)
plt.show()
fraud_user= df[df['isFraud']==1]['nameOrig'].value_counts()


fraud_type = df[df['type'].isin(['TRANSFER','CASH_OUT'])]

sns.countplot(data= fraud_type , x='type',hue='isFraud')
plt.title("Fraud Distribution in CASH OUT AND TRANSFER")
plt.grid(True)
plt.show()

# print(df.columns)
corr = df[['amount','oldbalanceOrg' ,'newbalanceOrig' ,'oldbalanceDest','newbalanceDest','isFraud']].corr()
# print(corr)

plt.figure(figsize=(8,6))
sns.heatmap(corr ,annot=True,cmap='coolwarm' , fmt='.2f')
plt.title("Correlation Matrix")
plt.show()