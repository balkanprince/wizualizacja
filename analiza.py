import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt

# Ustawienia
sns.set_style("darkgrid")

# Wczytanie danych
df = pd.read_csv(r'C:\Users\mpiesio\Desktop\KODILLA\wizualizacja\HRDataset.csv')

# Konwersje dat
df['DOB'] = pd.to_datetime(df['DOB'], format='%m/%d/%y', errors='coerce')
df['DateofHire'] = pd.to_datetime(df['DateofHire'], format='%m/%d/%Y', errors='coerce')
df['DateofTermination'] = pd.to_datetime(df['DateofTermination'], format='%m/%d/%y', errors='coerce')

# Obliczanie wieku i stażu pracy
df['Age'] = (dt.datetime(2019, 9, 27) - df['DOB']).apply(lambda x: x.days) / 365.25
df['Seniority'] = df.apply(
    lambda row: ((dt.datetime(2019, 9, 27) if pd.isnull(row['DateofTermination']) else row['DateofTermination']) - row['DateofHire']).days / 365.25,
    axis=1
)

df['HispanicLatino'] = df['HispanicLatino'].str.title()

# ========== ANALIZY ==========

# 1. 
plt.figure(figsize=(12, 5))
sns.boxplot(x='ManagerName', y='PerformanceScore', data=df)
plt.title("Performance Score vs Manager")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 2. 
plt.figure(figsize=(12, 5))
sns.boxplot(x='RecruitmentSource', y='Seniority', data=df)
plt.title("Seniority vs Recruitment Source")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 3. 
plt.figure(figsize=(10, 5))
sns.violinplot(x='MaritalDesc', y='EmpSatisfaction', data=df)
plt.title("Satisfaction vs Marital Status")
plt.tight_layout()
plt.show()

# 4. 
plt.figure(figsize=(8, 4))
sns.histplot(df['Age'].dropna(), bins=20)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# 5. 
plt.figure(figsize=(8, 4))
sns.lmplot(x='Age', y='SpecialProjectsCount', data=df, aspect=1.5)
plt.title("Special Projects vs Age")
plt.tight_layout()
plt.show()
