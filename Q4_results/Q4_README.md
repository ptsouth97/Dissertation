# Question 4 

## Overview
Are there correlations between supervisor behaviors and supervisee pass rates?

The application was tested using Python 3.6.6 running on Ubuntu and relies on:
* [Matplotlib](https://matplotlib.org/)
* [Numpy](http://www.numpy.org/)
* [Pandas](https://pandas.pydata.org/)
* [Scipy](https://www.scipy.org/)
* [Scikit-learn](https://scikit-learn.org/stable/)

## How it works

1. Export survey data from Qualtrics as a comma separated (.csv) file
2. Use Python to load .csv file into a Pandas dataframe
3. Drop survey responses that were not finished (i.e., 'Finished'==False)
4. Combine columns with text-based options. (i.e., include information written by respondents in the 'Other' columns)
5. Replace text with integers where appropriate. (e.g., convert choices like '0-20%' to 1, '21-40%' to 2, etc.)
6. Drop all the survey metadata. (e.g., IP address, time submitted, etc.)
7. Drop all non-supervisor behaviors (i.e., demographics)
8. Fill in zeroes for missing values (cannot calculate stats for blank cells)
9. Create a new column in the dataframe called 'pass rate'
10. 'Pass rate' is calculated as '100% fieldwork pass rate' / ('100% fieldwork candidates - 'Discontinued fieldwork')
11. Drop all NaN in the dataframe
12. Filter out responses that are impossible (e.g., respondent reported more passes than candidates resulting in a pass rate greater than 100%)
13. Make a list of the column heading needed for research question 4
14. Make a list of the supervision behaviors
15. Make a new dataframe to hold the results
16. Loop through each behavior in the list of supervision behaviors:
	1. Slice all rows for the current behavior and its corresponding pass rate
	2. Drop any NaNs
	3. Use Scipy stats to calculate the Spearman correlation between the behavior's survey responses and calculated pass rate
	4. Store the resulting rho and p-values in the results dataframe
	5. Use Scikit-learn to perform a linear regression on the same data
	6. Make a scatter plot of the data and include the linear regression
