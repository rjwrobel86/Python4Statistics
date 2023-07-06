#Pandas For Data Analysis 
#Import Pandas
import pandas as pd

#Find working directory
import os
os.getcwd()

#Read and Excel file and a CSV file into two separate data frames
df1 = pd.read_csv('data.csv')
df2 = pd.read_excel('data.xlsx')

#View the first 5 rows of df1, the first 10 rows of df1, and the last 10 rows of df2
print(df1.head()) #5 is the default
print(df1.head(10))
print(df2.tail(10))

#Summary statistics
df1.describe()

#Count missing values and drop rows with missing values
print(df1.isnull().sum())

df1.dropna(inplace=True) # "inplace=True" modifies the existing data frame
print(df1.isnull().sum())

df1.describe()

#Check for duplicate rows
duplicate_count = df1.duplicated().sum()
print(duplicate_count)

#duplicates = df1.duplicated()
#print(duplicates)

#View instances of the duplicated rows
duplicated_rows = df1[df1.duplicated()]
duplicated_rows

#Checking for duplicates using a subset of columns instead of looking for exact duplicates
duplicates = df1.duplicated(subset=['Name', 'Age'])

#Drop duplicate rows
df1.drop_duplicates(inplace=True)

df1.columns

#Split "Text to columns"
df1[['First Name', 'Last Name']] = df1['Name'].str.split(n=1, expand=True)
df1.columns
df1.head()

#Renaming columns with a dictionary
df1.rename(columns={'Name': 'Full Name'}, inplace=True)
df1.head()

#Filter rows based on condition
over30 = df1[df1['Age'] > 30]
over30.head()

#Sorting by a column
age_desc = df1.sort_values('Age', ascending=False)
print(age_desc)

age_asc = df1.sort_values('Age', ascending=True)
print(age_asc)

#Merging data frames
dfL = pd.DataFrame({'ID': [1, 2, 3, 4], 'Value': ['A', 'B', 'C','D']})
dfR = pd.DataFrame({'ID': [3, 4, 5, 6], 'Value': ['C', 'D', 'E','F']})
print(dfL.head())
print(dfR.head())

#Inner join
inner_merged_df = pd.merge(dfL, dfR, on='ID', how='inner')
print(inner_merged_df)

#Outer join
outer_merged_df = pd.merge(dfL, dfR, on='ID', how="outer")
print(outer_merged_df)

#Left join
left_merged_df = pd.merge(dfL, dfR, on='ID', how="left")
print(left_merged_df)

#Right join
right_merged_df = pd.merge(dfL, dfR, on='ID', how="right")
print(right_merged_df)

#Aggregating / Grouping
#Group by category and get mean value and count
df_grouped = df1.groupby('Category').agg({'Value': 'mean', 'Count': 'sum'})
df_grouped

#Pivot tables
pivot_table = df1.pivot_table(values='Value', index='Category', columns='Count', aggfunc='sum')
print(pivot_table)

#Write a data frame to a CSV file
df_grouped.to_csv('output.csv', index=False)

#Write a data frame to an Excel file
df_grouped.to_excel('output.xlsx', index=False)