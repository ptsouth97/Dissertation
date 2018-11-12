#!/usr/bin/python3

import question_1, question_2, question_3, question_4, question_2_addendum

def main():
	''' Runs all 4 research questions at once'''

	print('NOW RUNNING RESEARCH QUESTION 1...')
	question_1.q1_prep()

	print('NOW RUNNING RESEARCH QUESTION 2...')
	question_2.q2_prep()

	print('NOW RUNNING RESEARCH QUESTION 2 ADDENDUM...')
	question_2_addendum.q2_add_prep()

	print('NOW RUNNING RESEARCH QUESTION 3...')
	question_3.q3_prep()

	print('NOW RUNNING RESEARCH QUESTION 4...')
	question_4.q4_prep()


if __name__ == '__main__':
	main()
