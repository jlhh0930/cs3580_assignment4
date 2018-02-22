import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multicomp import MultiComparison

print("SECTION ONE: ANOVAS")
print("ANOVA 1 - perform an ANOVA on 'Sex' column using 'Survived' as the independent variable.")
titanic = pd.read_csv('train.csv')
titanic.boxplot('Survived', by='Sex', figsize=(12,8))
plt.show()
model = ols('survived ~ sex', data=titanic).fit()
aov_table = sm.stats.anova_lm(model, typ=2)
print("ANOVA results:")
print(aov_table)


print("ANOVA 2 - perform an ANOVA on 'Pclass' column using 'Survived' as the independent variable.")
titanic = pd.read_csv('train.csv')
titanic.boxplot('Survived', by='Pclass', figsize=(12,8))
plt.show()
model = ols('survived ~ Pclass', data=titanic).fit()
aov_table = sm.stats.anova_lm(model, typ=2)
print("ANOVA results:")
print(aov_table)

print("SECTION TWO")
print("Correlation between female and survived")
print("Correlation between male and survived")
print("Pick two columns and visualize their distributions.  If they are not normal, transform.")
#distribution: sns.distplot(titanic['Survived'])

print("SECTION THREE: BIVARIATE VISUALIZATIONS")
print("Bivariate visualization between 'Survived' and 'Age'")
titanic = pd.read_csv('train.csv')
sns.distplot(titanic['Age'])
plt.show()
titanic.boxplot('Survived', by='Age', figsize=(12,8))
plt.show()
print("Bivariate visualization between 'Survived' and 'SibSp'")
titanic = pd.read_csv('train.csv')
sns.distplot(titanic['Age'])
plt.show()
titanic.boxplot('Survived', by='SibSp', figsize=(12,8))
plt.show()
print("Bivariate visualization between 'Survived' and 'Parch'")
titanic = pd.read_csv('train.csv')
sns.distplot(titanic['Parch'])
plt.show()
titanic.boxplot('Survived', by='Parch', figsize=(12,8))
plt.show()
print("Bivariate visualization between 'Survived' and 'Fare'")
titanic = pd.read_csv('train.csv')
sns.distplot(titanic['Fare'])
plt.show()
titanic.boxplot('Survived', by='Fare', figsize=(12,8))
plt.show()

print("SECTION FOUR: MULTIVARIATE VISUALIZATION")
print("Interaction between multiple variables.")
print("Heat map:")
print("Parallel Coordinates:")
print("Pair Plot:")

print("SECTION FIVE: REPORT")
