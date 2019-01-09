# Question 2

## Overview
Are there significant differences between the averages of reported frequency of individual supervisor behaviors relative to other individual supervisor behaviors?

The application was tested using Python 3.6.6 running on Ubuntu and relies on:
* [Matplotlib](https://matplotlib.org/)
* [Numpy](http://www.numpy.org/)
* [Pandas](https://pandas.pydata.org/)
* [Scipy](https://www.scipy.org/)

## How it works

1. Export survey data from Qualtrics as a comma separated (.csv) file
2. Use Python to load .csv file into a Pandas dataframe
3. Drop survey responses that were not finished (i.e., 'Finished'==False)
4. Combine columns with text-based options. (i.e., include information written by respondents in the 'Other' columns)
5. Replace text with integers where appropriate. (e.g., convert choices like '0-20%' to 1, '21-40%' to 2, etc.)
6. Drop all the survey metadata. (e.g., IP address, time submitted, etc.)
7. Drop non-supervisor behaviors (i.e., demographics)
8. For each column (which at this point only represent supervisor behaviors) in the dataframe, drop the NaNs and turn the column into a list
9. Pass all the lists (representing all the supervisor behaviors) to scipy stats' oneway ANOVA which returns F and p-value
10. Calculate the degrees of freedom based on the number of groups and the number of items in each group
11. Use degrees of freedom to calculate the critical F-value
12. Make a labeled boxplot of all the behaviors
