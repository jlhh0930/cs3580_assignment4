import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multicomp import MultiComparison

print("SECTION ONE: ANOVAS")
print("ANOVA 1 - perform an ANOVA on 'Sex' column using 'Survived' as the independent variable.")
#get the data into a dataframe
dir = os.path.dirname(__file__)
filename = os.path.join(dir, 'train.csv')
titanic = pd.read_csv(filename)
#set the PassengerId as the index
titanic = titanic.set_index('PassengerId')
print('I changed the values in the Sex column to represent Females with 0 and Males with 1.')
#titanic['sex_int'] = NaN
def transform_sex_to_int(value):
    if value == 'male':
        return 1
    elif value == 'female':
        return 0
titanic['sex_int'] = titanic[['Sex']].applymap(transform_sex_to_int)

#show descriptive statistics for the dataframe
print('The following table is a set of descriptive statistics for the data in train.csv.')
print('This helps me see if the results in my visualizations make sense.')
print(titanic.describe(include=['number']))

print('Perform ANOVA on Survived and Sex.')
print('Because I changed the values of male and female in a separate column, I ran the stats on sex_int.')
print('First, I wanted to see a barchart or count plot of survival by sex.')
print('This count plot shows female as 0, male as 1, and shows the count of deaths.')
print('As we see, more men died than women.')
fig, axes = plt.subplots(1, 1, figsize=(16,10))
sns.countplot('sex_int',data=titanic)
plt.show()
print('Next, the ANOVA on Survived and sex_int.')
model = ols('Survived ~ sex_int', data=titanic).fit()
aov_table = sm.stats.anova_lm(model, typ=2)
print("ANOVA results:")
print(aov_table)
print('What these results mean:')
print('TODO')

print("ANOVA 2 - perform an ANOVA on 'Pclass' column using 'Survived' as the independent variable.")
print('I wanted to see how many people survived by Pclass.')
print('First, I wanted to see a barchart or count plot of survival by Pclass.')
print('This count plot shows the count of deaths by Pclass.')
print('As we see, you were more likely to die if you were in Pclass 3.')
fig, axes = plt.subplots(1, 1, figsize=(16,10))
sns.countplot('Pclass', data=titanic)
plt.show()
model = ols('Survived ~ Pclass', data=titanic).fit()
aov_table = sm.stats.anova_lm(model, typ=2)
print("ANOVA results:")
print(aov_table)
print('What these results mean:')
print('TODO')

#print("SECTION TWO")
#print("Correlation between female and survived")
#print('Again, I used my created sex_int column where females are represented as 0, males as 1.')
#print('To complete this correlation, I needed to delete all records for males.')
##delete unneeded columns
#female_survived = titanic
#del female_survived['Sex']
#del female_survived['Pclass']
#del female_survived['Age']
#del female_survived['SibSp']
#del female_survived['Parch']
#del female_survived['Ticket']
#del female_survived['Fare']
#del female_survived['Cabin']
#del female_survived['Embarked']
#female = (female_survived['sex_int'] == 0)
#female_survived = female_survived[female]
#correlation_female_survived = female_survived.corr(method='pearson')
#print(correlation_female_survived)
#print("Correlation between male and survived")
#print('Again, I used my created sex_int column where females are represented as 0, males as 1.')
#print('To complete this correlation, I needed to delete all records for females.')
##delete unneeded columns
#male_survived = titanic
#del male_survived['Sex']
#del male_survived['Pclass']
#del male_survived['Age']
#del male_survived['SibSp']
#del male_survived['Parch']
#del male_survived['Ticket']
#del male_survived['Fare']
#del male_survived['Cabin']
#del male_survived['Embarked']
#male = (male_survived['sex_int'] == 1)
#male_survived = male_survived[male]
#correlation_male_survived = male_survived.corr(method='pearson')
#print(correlation_male_survived)
print("Pick two columns and visualize their distributions.  If they are not normal, transform.")
print('Distribution of Age:')
sns.distplot(titanic['Age'].dropna(),kde=True)
plt.show()
print('Distribution of Fare:')
sns.distplot(titanic['Fare'].dropna(),kde=True)
plt.show()
print('TODO: CHECK FOR NORMALITY')


print("SECTION THREE: BIVARIATE VISUALIZATIONS")
print("Bivariate visualization between 'Survived' and 'Age'")
sns.boxplot(x="Survived", y="Age", data=titanic)
plt.show()
print("Bivariate visualization between 'Survived' and 'SibSp'")
titanic.groupby('SibSp')['Survived'].mean().plot(kind='barh')
plt.show()
print("Bivariate visualization between 'Survived' and 'Parch'")
titanic.groupby('Parch')['Survived'].mean().plot(kind='barh')
plt.show()
print("Bivariate visualization between 'Survived' and 'Fare'")
sns.boxplot(x="Survived", y="Fare", data=titanic)
plt.show()
print("Bivariate visualization between 'Survived' and 'Pclass'")
titanic.groupby('Pclass')['Survived'].mean().plot(kind='barh')
plt.show()

print("SECTION FOUR: MULTIVARIATE VISUALIZATION")
print("Interaction between multiple variables.")
print("Heat map:")
f, ax = plt.subplots(figsize=(10,8))
corr = titanic.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True), ax=ax)
plt.show()

print("SECTION FIVE: REPORT")
