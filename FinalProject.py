import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

df = pd.read_csv('Billionaires Statistics Dataset.csv', 
                 usecols = ['worth', 'sector', 'personName',
                            'age', 'country', 'company', 'selfMade',
                            'gender', 'birthYear'])
print(df.info)

print(df.size)

print("here")
print(df.notna)

#checking null values
df.isnull().sum()

#drop Na values
df = df.dropna()

#data types of columns
df.dtypes

#describe birth year and worth
df[['worth', 'age']].describe()


#uniques values in sector
df['sector'].unique()

#rows 
df.count()

#columns
len(df.columns)

#info
df.info()

#shape
df.shape

#first rows
df.head()

#renaming columns
df.rename(columns = {'personName' : 'Name', 
                     'birthYear' : 'date of birth'})

# sector worth (Bar chart)
# What are the top 5 sectors worth
sector_worth = pd.DataFrame()
sector_worth = df.groupby('sector')['worth'].sum().reset_index()
sector_worth.columns = ['Sector', 'Total Worth']

sector_worth = sector_worth.sort_values(by='Total Worth', ascending=False)

plt.figure(figsize = (10,5))
plt.bar(sector_worth['Sector'][0:5], sector_worth['Total Worth'][0:5])  
plt.xlabel('Top 5 Sectors')  
plt.ylabel('Worth')
plt.show()

# age and worth (scatter plot)
# What age group has the most worth amoung billionaires?
age_worth = pd.DataFrame()
age_worth = df.groupby('age')['worth'].sum().reset_index()
age_worth.columns = ['Age', 'Worth']

plt.scatter(age_worth['Age'], age_worth['Worth'], c = 'red')
plt.xlabel('Age')  
plt.ylabel('Worth')
plt.show()

# gender and sector (females arered males are blue)
# What is the difference between male and female worth amoung sectors?
gender_sector = df.groupby(['sector', 'gender']).size().reset_index(name='count')
gender_sector = gender_sector.pivot_table(index='sector', columns='gender', values='count', fill_value=0)

gender_sector.reset_index(inplace=True)
X_axis = np.arange(len(gender_sector['sector'])) 
plt.bar(X_axis - 0.2, gender_sector['F'], 0.4, label = 'Females', color = 'red')
plt.bar(X_axis, gender_sector['M'], 0.4, label = 'Males', color = 'blue')
plt.xticks(X_axis, gender_sector['sector'], rotation = 90) 
plt.xlabel("Sectors")
plt.show()

# Self made vs not self made worth
# What billionaires are worth more on average?
self_made = df.groupby(['selfMade', 'worth'])
sns.boxplot(x='selfMade', y='worth', data=df, palette='Set2')
plt.xlabel('Selfmade')
plt.ylabel('Net Worth')
plt.show()

# What group of billionaires is worth worth more?
worth = df.groupby('selfMade')['worth'].sum().reset_index()
plt.bar(worth['selfMade'], worth['worth'])
plt.xlabel('Selfmade')
plt.ylabel('Net Worth')
plt.show()

# What are the names of the top 5 billionaires by worth?
df = df.sort_values(by = 'worth', ascending = False)
print(df.head(5)[['personName', 'worth']])

# Average of billionaire wealth of each country(top5)?
country_worth = pd.DataFrame()
country_worth = df.groupby('country')['worth'].mean()
country_worth = country_worth.sort_values(ascending=False)
print(country_worth.head(5))

# What is the most commmon sector amoung billionaires?
common_industry = df['sector'].value_counts().idxmax()
print("the most popular sector is", (common_industry))

#using a chi square test to determine if there is a correlation between
#the gender of billionaires and there country
contingency_table = pd.crosstab(df['gender'], df['country'])
chi2, p_val, _, _ = stats.chi2_contingency(contingency_table)
print(f"Chi-square value: {chi2}")
print(f"P-value: {p_val}")

print('According to the chi square value it suggests that there is a difference between the observation and expected distribution')
print('According to the the pvalue there is no statistical relationship between the gender of billionaires and the countries they come from')

