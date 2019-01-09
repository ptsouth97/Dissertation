# Question 3

## Overview
Are there significant differences between frequency of individual supervisor behaviors relative to supervisor demographics?

The application was tested using Python 3.6.6 running on Ubuntu and relies on:
* [Matplotlib](https://matplotlib.org/)
* [Numpy](http://www.numpy.org/)
* [Pandas](https://pandas.pydata.org/)
* [Scipy](https://www.scipy.org/)

## How it works

1. Export survey data from Qualtrics as a comma separated (.csv) file
2. Use Python to load the .csv file into a Pandas dataframe
3. Drop survey responses that were not finished (i.e., 'Finished'==False)
4. Combine columns with text-based options. (i.e., include information written by respondents in the 'Other' columns)
5. Replace text with integers where appropriate. (e.g., convert choices like '0-20%' to 1, '21-40%' to 2, etc.)
6. Drop all the survey metadata. (e.g., IP address, time submitted, etc.)
7. Make a list of the demographics
8. Make a list of the supervision behaviors
9. Make a new dataframe to hold the results
10. Loop through list of demographics
	1. Replace blank cells with NaN
	2. Drop all NaN cells
	3. For each demographic, loop through the list of supervision behaviors
	4. Slice the dataframe column of the current supervision behavior 
	5. Pass the data as lists to Scipy stats one-way ANOVA function which returns F and p-value
	6. Use degrees of freedom to calculate critical F-value
	7. Display the results as a boxplot
	8. Continue with each behavior for this demographic, then move to the next demographic and repeat
