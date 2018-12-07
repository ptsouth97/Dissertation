# Question 4 

## Overview
BONUS: How accurately can a basic k-NN model classify the set of survey responses by demographic?

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
4. Drop the survey responses that did not meet BACB supervision requirements
5. Combine columns with text-based options. (i.e., include information written by respondents in the 'Other' columns)
6. Replace text with integers where appropriate. (e.g., convert choices like '0-20%' to 1, '21-40%' to 2, etc.)
7. Drop all the survey metadata. (e.g., IP address, time submitted, etc.)
8. Make a list of the demographics
9. Make a list of the supevision behaviors
10. Pass the two previous lists along with the data frame to the question5 function
11. Intiate another dataframe to hold the knn score results
12. Loop through the list of demographics
	1. Count the number of occurences of unique answers for the current demographic
	2. Drop row that have counts less than 2 (knn won't work if < 2)
	3. Create the target array (y)
	4. Use get_dummies(y) to encode categorical data of the demographic (State, Area of study, etc.)
	5. Create the feature array (X)
	6. Drop columns that don't contain data from X
	7. Convert X to datatype 'int64' (float)
	8. Split X and y into training and test sets
	9. Instantiate a k-NN classifier
	10. Fit the classifier to the training data
	11. Score the accuracy
	12. Record the results in the dataframe
	13. Return to the main function
13. Display results as a graph
	
