#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt


def main():
	''' main function for testing'''

	points, new_points = make_points()
	

def plot_clusters(points):
	''' displays clusters on scatter plot'''

	# Slice the first and second columns
	xs = points[:, 0]
	ys = points[:,1]

	
	xA = np.random.normal(xs, 0.1, len(xs))
	yA = np.random.normal(ys, 0.1, len(ys))

	# 'Jitter' the points so they do not overlap
	#yA = np.random.normal(1, 0.1, len(ys))
	#xA = np.random.normal(1, 0.1, len(ys))
	
	fig = plt.figure()
	plt.scatter(xA, yA, alpha=0.5,  color='black')
	plt.xlabel('Confirm required skill set survey responses')
	plt.ylabel('Literature for new competency survey responses')
	plt.title('Example of clustering for 2 of 46 dimensions')
	caption = '$\it{Fig 1.}$ Survey responses for two separate supervisor behaviors plotted against each other show. The responses are jittered to show multiple responses for the same value that would otherwise overlap'
	fig.text(0.01, 0.01, caption, ha='left', wrap=True)
	plt.savefig('ExampleClustering.png')
	plt.show()
 
	return


if __name__ == '__main__':
	main()
