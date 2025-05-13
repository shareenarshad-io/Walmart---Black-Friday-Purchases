'''
Objective
This notebook aims to analyze the customer purchase behaviour (specifically, purchase amount) against the customer’s gender and the various other factors to help the business make better decisions. The Management team at Walmart Inc. wants to understand if the spending habits differ between male and female customers:

Do women spend more on Black Friday than men?
'''

#Exploratory Data Analysis

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes = True)
import scipy.stats as stats
from scipy.stats import norm
import warnings
warnings.filterwarnings("ignore")
walmart_df = pd.read_csv("walmart_data.csv")

#Data Overview
#Here, we'll first look at the Walmart dataset. We'll check the first and last few rows to understand what the data looks like.

walmart_df.head()

walmart_df.tail()

# Shape of the dataframe
walmart_df.shape
# Name of each column in dataframe
walmart_df.columns
# Datatype of each column in dataframe
walmart_df.dtypes


#Finding Unique Values and Count Them

def print_unique_values(df):
    for column in df.columns:
        unique_values = df[column].unique()
        print(f"\nUnique Values of {column}: ", unique_values)

# Example usage
print_unique_values(walmart_df)


def print_nunique_values(df):
    for column in df.columns:
        unique_values = df[column].nunique()
        print(f"\nUnique Values of {column}: ", unique_values)

# Example usage
print_nunique_values(walmart_df)

#Data Cleaning

walmart_df.Stay_In_Current_City_Years.unique()
# Removing "+" symbol
walmart_df.Stay_In_Current_City_Years=walmart_df.Stay_In_Current_City_Years.str.replace("+","")
walmart_df.Stay_In_Current_City_Years.unique()
# Assuming '+' symbols have already been removed as per your screenshot
walmart_df['Stay_In_Current_City_Years'] = pd.to_numeric(walmart_df['Stay_In_Current_City_Years'])

#Statistical Summary
walmart_df.select_dtypes(include=['int64']).skew()
walmart_df.describe(include = 'all')

'''
We'll share some key observations from the analysis. This could include things like which age group buys the most, which city has the most sales, etc.

Observations from Analysis
There are no missing values in the data.

Customers with age group of 26-35 have done more purchases (2,19,587) compared with others

Customers in City_Category of B have done more purchases (2,31,173) compared with other City_Category
Out of 5,50,000 data point. 4,14,259's gender is Male and rest are the Female.
Customer with Minimum amount of Purchase is 12 $

Customer with Maximum amount of Purchase is 23961 $

Purchase might have outliers

Missing Value Detection
We'll check if there are any missing values in the data. Missing values can affect the analysis, so it's important to know if there are any. So Let's check the missing values in our dataset.
'''

# Missing value detection
walmart_df.isna().sum()

#Duplicate Value Detection

# Checking duplicate values in the data set
walmart_df.duplicated(subset=None,keep='first').sum() # No duplicate values in the data set

#Data Visualization
walmart_df.info()

#Data Visualization with Numerical Features

[col for col in walmart_df.select_dtypes(include=['int64']).columns]

import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm

# Create a 2x2 grid of subplots
fig, axis = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
fig.subplots_adjust(top=0.9)  # Adjust the top spacing of the subplots

# Plot distribution plots for each specified column
sns.distplot(walmart_df['Occupation'], kde=True, ax=axis[0,0], color="#900000")
sns.distplot(walmart_df['Stay_In_Current_City_Years'].astype(int), kde=True, ax=axis[0,1], color="#900000")
sns.distplot(walmart_df['Marital_Status'], kde=True, ax=axis[1,0], color="#900000")

# Plotting a distribution plot for the 'Purchase' variable with normal curve fit
sns.distplot(walmart_df['Purchase'], ax=axis[1,1], color="#900000", fit=norm)

# Fitting the target variable to the normal curve 
mu, sigma = norm.fit(walmart_df['Purchase']) 
print("The mu (mean) is {} and sigma (standard deviation) is {} for the curve".format(mu, sigma))

# Adding a legend for the 'Purchase' distribution plot
axis[1,1].legend(['Normal Distribution (μ = {:.2f}, σ = {:.2f})'.format(mu, sigma)], loc='best')

# Show the plots
plt.show()

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Create subplots
fig = make_subplots(
    rows=4, cols=2,
    subplot_titles=("Gender", "Age", "Occupation", "City Category",
                    "Stay In Current City Years", "Marital Status", "Product Category", "Purchase")
)

# Add histograms for each subplot
fig.add_trace(go.Histogram(x=walmart_df['Gender']), row=1, col=1)
fig.add_trace(go.Histogram(x=walmart_df['Age']), row=1, col=2)
fig.add_trace(go.Histogram(x=walmart_df['Occupation']), row=2, col=1)
fig.add_trace(go.Histogram(x=walmart_df['City_Category']), row=2, col=2)
fig.add_trace(go.Histogram(x=walmart_df['Stay_In_Current_City_Years']), row=3, col=1)
fig.add_trace(go.Histogram(x=walmart_df['Marital_Status']), row=3, col=2)
fig.add_trace(go.Histogram(x=walmart_df['Product_Category']), row=4, col=1)
fig.add_trace(go.Histogram(x=walmart_df['Purchase']), row=4, col=2)

# Update layout if needed
fig.update_layout(height=1200, width=1000, title_text="Count Plots")
fig.update_layout(showlegend=False)  # Hide the legend if not needed

# Show the figure
fig.show()

'''
Observations:

Many buyers are male while the minority are female. Difference is due to the categories on sale during Black Friday, evaluating a particular category may change the count between genders.

There are 7 categories defined to classify the age of the buyers

Majority of the buyers are single

Display of the occupation of the buyers. Occupation 8 has extremely low count compared with the others; it can be ignored for the calculation since it won't affect much the result.

Majority of the products are in category 1, 5 and 8. The low number categories can be combined into a single category to greatly reduce the complexity of the problem.

Higher count might represent the urban area indicates more population in City_Category.

Most buyers have one year living in the city. Remaining categories are in uniform distribution

Data Visualization with Categorical Features
Here, we'll focus on the categorical data, like gender, age, and city category. We'll use different types of charts to show how these categories relate to purchases. This will help us understand which categories have the most impact on purchasing behavior.
'''
fig, axis = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
fig.subplots_adjust(top=1.2)

sns.boxplot(data=walmart_df, x="Occupation", ax=axis[0,0])
sns.boxplot(data=walmart_df, x="Stay_In_Current_City_Years", orient='h', ax=axis[0,1])
sns.boxplot(data=walmart_df, x="Purchase", orient='h', ax=axis[1,0])


plt.show()

#Purchase & Our Features

attrs = ['Gender', 'Age', 'Occupation', 'City_Category', 'Stay_In_Current_City_Years', 'Marital_Status', 'Product_Category']
sns.set(color_codes = True)
fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(20, 16))
fig.subplots_adjust(top=1.3)
count = 0
for row in range(3):
    for col in range(2):
        sns.boxplot(data=walmart_df, y='Purchase', x=attrs[count], ax=axs[row, col])
        axs[row,col].set_title(f"Purchase vs {attrs[count]}", pad=12, fontsize=13)
        count += 1
plt.show()

plt.figure(figsize=(10, 8))
sns.boxplot(data=walmart_df, y='Purchase', x=attrs[-1])
plt.show()

sns.set(color_codes = True)
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(20, 6))

fig.subplots_adjust(top=1.5)
sns.boxplot(data=walmart_df, y='Purchase', x='Gender', hue='Age', ax=axs[0,0])
sns.boxplot(data=walmart_df, y='Purchase', x='Gender', hue='City_Category', ax=axs[0,1])

sns.boxplot(data=walmart_df, y='Purchase', x='Gender', hue='Marital_Status', ax=axs[1,0])
sns.boxplot(data=walmart_df, y='Purchase', x='Gender', hue='Stay_In_Current_City_Years', ax=axs[1,1])
axs[1,1].legend(loc='upper left')

plt.show()

#Data Analysis
#1. Are women spending more money per transaction than men? Why or Why not?

# Average amount spend per customer for Male and Female
amt_df = walmart_df.groupby(['User_ID', 'Gender'])[['Purchase']].sum()
avg_amt_df = amt_df.reset_index()
avg_amt_df


# Gender wise value counts in avg_amt_df
avg_amt_df['Gender'].value_counts()


# Gender wise value counts in avg_amt_df
avg_amt_df['Gender'].value_counts()

# histogram of average amount spend for each customer - Female
avg_amt_df[avg_amt_df['Gender']=='F']['Purchase'].hist(bins=100)
plt.show()

male_avg = avg_amt_df[avg_amt_df['Gender']=='M']['Purchase'].mean()
female_avg = avg_amt_df[avg_amt_df['Gender']=='F']['Purchase'].mean()

print("Average amount spend by Male customers: {:.2f}".format(male_avg))
print("Average amount spend by Female customers: {:.2f}".format(female_avg))

#2. Confidence intervals and distribution of the mean of the expenses by female and male customers

male_df = avg_amt_df[avg_amt_df['Gender']=='M']
female_df = avg_amt_df[avg_amt_df['Gender']=='F']

genders = ["M", "F"]

male_sample_size = 3000
female_sample_size = 1500
num_repitions = 1000
male_means = []
female_means = []

for _ in range(num_repitions):
    male_mean = male_df.sample(male_sample_size, replace=True)['Purchase'].mean()
    female_mean = female_df.sample(female_sample_size, replace=True)['Purchase'].mean()
    
    male_means.append(male_mean)
    female_means.append(female_mean)

fig, axis = plt.subplots(nrows=1, ncols=2, figsize=(20, 6))

axis[0].hist(male_means, bins=100)
axis[1].hist(female_means, bins=100)
axis[0].set_title("Male - Distribution of means, Sample size: 9500")
axis[1].set_title("Female - Distribution of means, Sample size: 8500")

plt.show()

print("\n")
print("Population mean - Mean of sample means of amount spend for Male: {:.2f}".format(np.mean(male_means)))
print("Population mean - Mean of sample means of amount spend for Female: {:.2f}".format(np.mean(female_means)))

print("\nMale - Sample mean: {:.2f} Sample std: {:.2f}".format(male_df['Purchase'].mean(), male_df['Purchase'].std()))
print("Female - Sample mean: {:.2f} Sample std: {:.2f}".format(female_df['Purchase'].mean(), female_df['Purchase'].std()))

male_margin_of_error_clt = 1.64*male_df['Purchase'].std()/np.sqrt(len(male_df))
male_sample_mean = male_df['Purchase'].mean()
male_lower_lim = male_sample_mean - male_margin_of_error_clt
male_upper_lim = male_sample_mean + male_margin_of_error_clt

female_margin_of_error_clt = 1.64*female_df['Purchase'].std()/np.sqrt(len(female_df))
female_sample_mean = female_df['Purchase'].mean()
female_lower_lim = female_sample_mean - female_margin_of_error_clt
female_upper_lim = female_sample_mean + female_margin_of_error_clt

print("Male confidence interval of means: ({:.2f}, {:.2f})".format(male_lower_lim, male_upper_lim))
print("Female confidence interval of means: ({:.2f}, {:.2f})".format(female_lower_lim, female_upper_lim))

male_margin_of_error_clt = 1.96*male_df['Purchase'].std()/np.sqrt(len(male_df))
male_sample_mean = male_df['Purchase'].mean()
male_lower_lim = male_sample_mean - male_margin_of_error_clt
male_upper_lim = male_sample_mean + male_margin_of_error_clt

female_margin_of_error_clt = 1.96*female_df['Purchase'].std()/np.sqrt(len(female_df))
female_sample_mean = female_df['Purchase'].mean()
female_lower_lim = female_sample_mean - female_margin_of_error_clt
female_upper_lim = female_sample_mean + female_margin_of_error_clt

print("Male confidence interval of means: ({:.2f}, {:.2f})".format(male_lower_lim, male_upper_lim))
print("Female confidence interval of means: ({:.2f}, {:.2f})".format(female_lower_lim, female_upper_lim))
male_margin_of_error_clt = 2.58*male_df['Purchase'].std()/np.sqrt(len(male_df))
male_sample_mean = male_df['Purchase'].mean()
male_lower_lim = male_sample_mean - male_margin_of_error_clt
male_upper_lim = male_sample_mean + male_margin_of_error_clt

female_margin_of_error_clt = 2.58*female_df['Purchase'].std()/np.sqrt(len(female_df))
female_sample_mean = female_df['Purchase'].mean()
female_lower_lim = female_sample_mean - female_margin_of_error_clt
female_upper_lim = female_sample_mean + female_margin_of_error_clt

print("Male confidence interval of means: ({:.2f}, {:.2f})".format(male_lower_lim, male_upper_lim))
print("Female confidence interval of means: ({:.2f}, {:.2f})".format(female_lower_lim, female_upper_lim))

#3. Are confidence intervals of average male and female spending overlapping? How can Walmart leverage this conclusion to make changes or improvements?

'''
The confidence intervals of average male and female spendings are not overlapping.

Walmart can leverage this problem by taking sample dataset and apply this to whole population dataset by performing Central Limit Theorem and Confidence Intervals of 90%, 95%, or 99% by playing around with the width parameter by reporting those observations to Walmart.

'''

#4. Results when the same activity is performed for Married vs Unmarried

amt_df = walmart_df.groupby(['User_ID', 'Marital_Status'])[['Purchase']].sum()
avg_amt_df = amt_df.reset_index()
avg_amt_df

avg_amt_df['Marital_Status'].value_counts()

married_samp_size = 3000
married_samp_size = 2000
num_repitions = 1000
married_means = []
unmarried_means = []

for _ in range(num_repitions):
    married_mean = avg_amt_df[avg_amt_df['Marital_Status']==1].sample(married_samp_size, replace=True)['Purchase'].mean()
    unmarried_mean = avg_amt_df[avg_amt_df['Marital_Status']==0].sample(married_samp_size, replace=True)['Purchase'].mean()
    
    married_means.append(married_mean)
    unmarried_means.append(unmarried_mean)
    
    
fig, axis = plt.subplots(nrows=1, ncols=2, figsize=(20, 6))

axis[0].hist(married_means, bins=100)
axis[1].hist(unmarried_means, bins=100)
axis[0].set_title("Married - Distribution of means, Sample size: 3000")
axis[1].set_title("Unmarried - Distribution of means, Sample size: 2000")

plt.show()
print("\n")
print("Population mean - Mean of sample means of amount spend for Married: {:.2f}".format(np.mean(married_means)))
print("Population mean - Mean of sample means of amount spend for Unmarried: {:.2f}".format(np.mean(unmarried_means)))

print("\nMarried - Sample mean: {:.2f} Sample std: {:.2f}".format(avg_amt_df[avg_amt_df['Marital_Status']==1]['Purchase'].mean(), avg_amt_df[avg_amt_df['Marital_Status']==1]['Purchase'].std()))
print("Unmarried - Sample mean: {:.2f} Sample std: {:.2f}".format(avg_amt_df[avg_amt_df['Marital_Status']==0]['Purchase'].mean(), avg_amt_df[avg_amt_df['Marital_Status']==0]['Purchase'].std()))

married_samp_size = 3000
married_samp_size = 2000
num_repitions = 1000
married_means = []
unmarried_means = []

for _ in range(num_repitions):
    married_mean = avg_amt_df[avg_amt_df['Marital_Status']==1].sample(married_samp_size, replace=True)['Purchase'].mean()
    unmarried_mean = avg_amt_df[avg_amt_df['Marital_Status']==0].sample(married_samp_size, replace=True)['Purchase'].mean()
    
    married_means.append(married_mean)
    unmarried_means.append(unmarried_mean)
    
    
fig, axis = plt.subplots(nrows=1, ncols=2, figsize=(20, 6))

axis[0].hist(married_means, bins=100)
axis[1].hist(unmarried_means, bins=100)
axis[0].set_title("Married - Distribution of means, Sample size: 3000")
axis[1].set_title("Unmarried - Distribution of means, Sample size: 2000")

plt.show()
print("\n")
print("Population mean - Mean of sample means of amount spend for Married: {:.2f}".format(np.mean(married_means)))
print("Population mean - Mean of sample means of amount spend for Unmarried: {:.2f}".format(np.mean(unmarried_means)))

print("\nMarried - Sample mean: {:.2f} Sample std: {:.2f}".format(avg_amt_df[avg_amt_df['Marital_Status']==1]['Purchase'].mean(), avg_amt_df[avg_amt_df['Marital_Status']==1]['Purchase'].std()))
print("Unmarried - Sample mean: {:.2f} Sample std: {:.2f}".format(avg_amt_df[avg_amt_df['Marital_Status']==0]['Purchase'].mean(), avg_amt_df[avg_amt_df['Marital_Status']==0]['Purchase'].std()))


for val in ["Married", "Unmarried"]:
    
    new_val = 1 if val == "Married" else 0
    
    new_df = avg_amt_df[avg_amt_df['Marital_Status']==new_val] 
    
    margin_of_error_clt = 2.58*new_df['Purchase'].std()/np.sqrt(len(new_df))
    sample_mean = new_df['Purchase'].mean()
    lower_lim = sample_mean - margin_of_error_clt
    upper_lim = sample_mean + margin_of_error_clt

    print("{} confidence interval of means: ({:.2f}, {:.2f})".format(val, lower_lim, upper_lim))


#5. Results when the same activity is performed for Age

amt_df = walmart_df.groupby(['User_ID', 'Age'])[['Purchase']].sum()
avg_amt_df = amt_df.reset_index()
avg_amt_df

avg_amt_df['Age'].value_counts()


sample_size = 200
num_repitions = 1000

all_means = {}

age_intervals = ['26-35', '36-45', '18-25', '46-50', '51-55', '55+', '0-17']
for age_interval in age_intervals:
    all_means[age_interval] = []

for age_interval in age_intervals:
    for _ in range(num_repitions):
        mean = avg_amt_df[avg_amt_df['Age']==age_interval].sample(sample_size, replace=True)['Purchase'].mean()
        all_means[age_interval].append(mean)


for val in ['26-35', '36-45', '18-25', '46-50', '51-55', '55+', '0-17']:
    
    new_df = avg_amt_df[avg_amt_df['Age']==val] 
    
    margin_of_error_clt = 1.64*new_df['Purchase'].std()/np.sqrt(len(new_df))
    sample_mean = new_df['Purchase'].mean()
    lower_lim = sample_mean - margin_of_error_clt
    upper_lim = sample_mean + margin_of_error_clt
    
    print("For age {}, confidence interval of means: ({:.2f}, {:.2f})".format(val, lower_lim, upper_lim))

for val in ['26-35', '36-45', '18-25', '46-50', '51-55', '55+', '0-17']:
    
    new_df = avg_amt_df[avg_amt_df['Age']==val] 
    
    margin_of_error_clt = 1.96*new_df['Purchase'].std()/np.sqrt(len(new_df))
    sample_mean = new_df['Purchase'].mean()
    lower_lim = sample_mean - margin_of_error_clt
    upper_lim = sample_mean + margin_of_error_clt

    print("For age {}, confidence interval of means: ({:.2f}, {:.2f})".format(val, lower_lim, upper_lim))

for val in ['26-35', '36-45', '18-25', '46-50', '51-55', '55+', '0-17']:
    
    new_df = avg_amt_df[avg_amt_df['Age']==val] 
    
    margin_of_error_clt = 2.58*new_df['Purchase'].std()/np.sqrt(len(new_df))
    sample_mean = new_df['Purchase'].mean()
    lower_lim = sample_mean - margin_of_error_clt
    upper_lim = sample_mean + margin_of_error_clt

    print("For age {}, confidence interval of means: ({:.2f}, {:.2f})".format(val, lower_lim, upper_lim))

'''
Final Insights
After analyzing the data, we have gathered key insights about customer spending patterns based on age, gender, marital status, city category, and product categories.

Actionable Insights
For Age feature, we observed that ~ 80% of the customer's who belong to the age group 25-40 (40%: 26-35, 18%: 18-25, 20%: 36-45) tend to spend the most.

For Gender feature, ~75% of the number of purchases are made by Male customer's and rest of the 25% is done by female customer's. This tells us the Male consumers are the major contributors to the number of sales for the retail store.On average the male gender spends more money on purchase contrary to female, and it is possible to also observe this trend by adding the total value of purchase.

Average amount spend by Male customers: 9,25,408.28
Average amount spend by Female customers: 7,12,217.18
When we combined Purchase and Marital_Status for analysis (60% are Single, 40% are Married). We came to know that Single Men spend the most during the Black Friday. It also tells that Men tend to spend less once they are married. It maybe because of the added responsibilities.

There is an interesting column Stay_In_Current_City_Years, after analyzing this column we came to know the people who have spent 1 year in the city tend to spend the most. This is understandable as, people who have spent more than 4 years in the city are generally well settled and are less interested in buying new things as compared to the people new to the city, who tend to buy more (35% Staying in the city since 1 year, 18% since 2 years, 17% since 3 years).

When examining the City_Category which city the product was purchased to our surprise, even though the city B is majorly responsible for the overall sales income, but when it comes to the above product, it majorly purchased in the city C.

Total of 20 product_categories are there. Product_Category - 1, 5, 8, & 11 have highest purchasing frequency.

There are 20 differnent types of Occupation's in the city

Confidence Intervals
Now using the Central Limit Theorem for the population:

Average amount spend by male customers is 9,25,408.28
Average amount spend by female customers is 7,12,217.18
Now we can infer about the population that, 90% of the times:

Average amount spend by male customer will lie in between: (900471.15, 950217.65)
Average amount spend by female customer will lie in between: (679584.51, 744464.28)
Now we can infer about the population that, 95% of the times:

Average amount spend by male customer will lie in between: (895617.83, 955070.97)
Average amount spend by female customer will lie in between: (673254.77, 750794.02)
Now we can infer about the population that, 99% of the times:

Average amount spend by male customer will lie in between: (886214.53, 964474.27)
Average amount spend by female customer will lie in between: (660990.91, 763057.88)
Confidence Interval by Marital_Status

Now we can infer about the population that, 90% of the times:

Married confidence interval of means: (812686.46, 874367.13)
Unmarried confidence interval of means: (853938.67, 907212.90)
Now we can infer about the population that, 95% of the times:

Married confidence interval of means: (806668.83, 880384.76)
Unmarried confidence interval of means: (848741.18, 912410.38)
Now we can infer about the population that, 99% of the times:

Married confidence interval of means: (795009.68, 892043.91)
Unmarried confidence interval of means: (838671.05, 922480.51)
Confidence Interval by Age

Now we can infer about the population that, 90% of the times:

For age 26-35, confidence interval of means: (952320.12, 1026998.51)
For age 36-45, confidence interval of means: (832542.56, 926788.86)
For age 18-25, confidence interval of means: (810323.44, 899402.80)
For age 46-50, confidence interval of means: (726410.64, 858686.93)
For age 51-55, confidence interval of means: (703953.00, 822448.85)
For age 55+, confidence interval of means: (487192.99, 592201.50)
For age 0-17, confidence interval of means: (542553.13, 695182.50)
Now we can infer about the population that, 95% of the times:

For age 26-35, confidence interval of means: (945034.42, 1034284.21)
For age 36-45, confidence interval of means: (823347.80, 935983.62)
For age 18-25, confidence interval of means: (801632.78, 908093.46)
For age 46-50, confidence interval of means: (713505.63, 871591.93)
For age 51-55, confidence interval of means: (692392.43, 834009.42)
For age 55+, confidence interval of means: (476948.26, 602446.23)
For age 0-17, confidence interval of means: (527662.46, 710073.17)
Now we can infer about the population that, 99% of the times:

For age 26-35, confidence interval of means: (930918.39, 1048400.25)
For age 36-45, confidence interval of means: (805532.95, 953798.47)
For age 18-25, confidence interval of means: (784794.60, 924931.63)
For age 46-50, confidence interval of means: (688502.19, 896595.37)
For age 51-55, confidence interval of means: (669993.82, 856408.03)
For age 55+, confidence interval of means: (457099.09, 622295.40)
For age 0-17, confidence interval of means: (498811.78, 738923.84)
Recommendations
Men spent more money than women, So company should focus on retaining the female customers and getting more female customers.

Product_Category - 1, 5, 8, & 11 have highest purchasing frequency. it means these are the products in these categories are liked more by customers. Company can focus on selling more of these products or selling more of the products which are purchased less.

Unmarried customers spend more money than married customers, So company should focus on acquisition of married customers.

Customers in the age 25-40 spend more money than the others, So company should focus on acquisition of customers of other age groups.

The tier-2 city called B has the highest number of population, management should open more outlets in the tier-1 and tier-2 cities like A and C in order to increase the buisness.
'''
