# Question 1

## Overview
* Descriptive statistics of survey

The application was tested using Python 3.6.6 running on Ubuntu and relies on:
* [Matplotlib](https://matplotlib.org/)
* [Numpy](http://www.numpy.org/)
* [Pandas](https://pandas.pydata.org/)

## How it works

1. Export data from Qualtrics as a comma separated (.csv) file
2. Use Python to load .csv file into a Pandas dataframe
3. Drop survey responses that were not finished (i.e., 'Finished'==False)
4. Combine columns with text-based options. I.e., include information provided by respondents in the 'Other' columns
5. Replace text with integers where appropriate. E.g., convert choices like '0-20%' to 1, '21-40%' to 2, etc.
6. Drop all the survey metadata. E.g., IP address, time submitted, etc.
7. Make a list of all the demographics to be plotted
8. Loop through the list of demographics and:
	1. Replace blank cells with NaN
	2. Drop all cells that are NaN
	3. Slice the column of the current demographic
	4. Count the unique instances in the column
	5. Make a labeled barplot of the result
9. For the state demographic, make an additional plot that converts the number of raw responses into a percentage by dividing by the total number of BCBAs in each state
10. For questions that have multiple answers in one cell, separate those answers by splitting on the comma
11. Send the special cases through the same loop shown in number '8' above
