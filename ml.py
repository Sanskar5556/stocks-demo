import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

df=pd.read_csv("D:\Downloads\insurance.csv")

#EDA
print(df.shape)
print(df.head)
print(df.info())
print(df.describe())
print(df.isnull().sum())

print(df.columns)
numeric_col=["age","bmi","children","charges"]
for col in numeric_col:
     plt.figure(figsize=(6,4))
     sns.histplot(df[col],kde=True,bins=20)
     plt.show()
     plt.close()

sns.countplot(x=df["children"])
plt.show()
sns.countplot(x=df["sex"])
plt.show()
sns.countplot(x=df["smoker"])
plt.show()

for col in numeric_col:
    plt.figure(figsize=(6,4))
    sns.boxplot(x=df[col])
    plt.show()
    plt.close()

plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True),annot=True)
plt.show()

#Data cleaning and preprocessing
df_clean=df.copy()
df_clean.head()
print(df_clean.shape)
df_clean.drop_duplicates(inplace=True)
print(df_clean.shape)
print(df.isnull().sum())
print(df_clean.dtypes)
print(df_clean["sex"].value_counts())

df_clean["sex"]=df_clean["sex"].map({"male":1,"female":0})
df_clean["smoker"]=df_clean["smoker"].map({"yes":1,"no":0})
df_clean.rename(columns={"sex":"is_female","smoker":"is_smoker"},inplace=True)


print(df["region"].value_counts())
df_clean=pd.get_dummies(df_clean,columns=["region"],drop_first=True)
for col in ["region_northwest", "region_southwest", "region_southeast", "region_northeast"]:
    if col in df_clean.columns:
        df_clean[col] = df_clean[col].astype(int)

#Feature Enginnering and extraction
sns.histplot(df["bmi"])
plt.show()

df_clean["bmi_category"]=pd.cut(
    df_clean["bmi"],bins=[0,18.5,24.9,29.9,float("inf")],labels=["Underweight","Normal","Overweight","Obese"]
)

df_clean=pd.get_dummies(df_clean,columns=["bmi_category"],drop_first=True)
for col in ["bmi_category_Underweight","bmi_category_Normal", "bmi_category_Overweight", "bmi_category_Obese"]:
    if col in df_clean.columns:
        df_clean[col] = df_clean[col].astype(int)

from sklearn.preprocessing import StandardScaler
cols=["age","bmi","children"]
scaler=StandardScaler()
df_clean[cols]=scaler.fit_transform(df_clean[cols])

from scipy.stats import pearsonr
selected_features=["age","bmi","children","is_female","is_smoker","region_northwest","region_southeast","region_southwest",
"bmi_category_Normal","bmi_category_Overweight","bmi_category_Obese"]
correlations={
    feature:pearsonr(df_clean[feature],df_clean["charges"])[0]
    for feature in selected_features
    }

correlations_df=pd.DataFrame(list(correlations.items()),columns=["Feature","Pearson correlation"])
print(correlations_df.sort_values (by = "Pearson correlation",ascending=False))

cat_features=["is_female","is_smoker","region_northwest","region_southeast","region_southwest","bmi_category_Normal","bmi_category_Obese","bmi_category_Overweight"]

from scipy.stats import chi2_contingency
alpha=0.05
df_clean["charges_bins"]=pd.qcut(df_clean["charges"],q=4,labels=False)
chi2_results={}
for col in cat_features:
    contingency=pd.crosstab(df_clean[col],df_clean["charges_bins"])
    chi2_stat,p_val,_,_=chi2_contingency(contingency)
    decision="Reject null (keep feature)"if p_val<alpha else "Accept null (Drop Feature)"
    chi2_results[col]={
        "chi2_statistic":chi2_stat,
        "p_value":p_val,
        "Decision":decision
    }
chi2_df=pd.DataFrame(chi2_results).T
chi2_df=chi2_df.sort_values(by="p_value")
print(chi2_df)


