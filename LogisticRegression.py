import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from tensorflow.python import debug as tf_debug

from sklearn.linear_model import LinearRegression
from sklearn import linear_model

import pandas.io.common

class LogRegression():
	
	def __init__(self, path=None):

		try:
			self._df = pd.read_csv(path, header=None, delimiter=',')
		except pandas.io.common.EmptyDataError:
			print("No file")

		x_1 = self._df.iloc[:,0]
		x_2 = self._df.iloc[:,1]

		self._l = len(x_1)
		ones = np.ones((self._l, 1))
		#self.x = np.append(np.ones((1,2)), np.column_stack((x_1,x_2)), axis=0)
		self.x = np.c_[ones, x_1, x_2]
		self.y = self._df.iloc[:,2]
		#self._l = len(self._df.iloc[:,0])

	def sygmoid(self, data = None):

		if (data is None):
			data = self.x
		g_z = 1 / (1 + np.exp(data))
		return (g_z)

	def costFunction(self, thetta=np.zeros((3, 1))):
		
		#thetta = np.zeros((3, 1))
		ones = np.ones((self._l))
		hypotesis = self.sygmoid(np.dot(self.x, thetta))
		log_hypotesis = np.log(hypotesis)
		log_hypotesis_2 = np.log(np.subtract(ones, hypotesis))
		first_part = np.dot(np.dot(self.y, -1), log_hypotesis)
		second_part = np.dot((np.subtract(ones, self.y)), log_hypotesis_2)

		list_ = np.subtract(first_part, second_part)
		J_cost = (1 / self._l) * np.sum(list_)
		return (J_cost)
		
	def gradient(self, thetta=np.zeros((3, 1))):

		hypotesis = self.sygmoid(np.dot(self.x, thetta))
		hypotesis = np.reshape(hypotesis, self._l)
		list_ = np.dot(np.subtract(hypotesis, self.y), self.x)
		sum_ = np.sum(list_)

		return ((1 / self._l) * sum_)


	def plotData(self):

		plt.figure(1)
		plt.axis([30, 100, 30, 100])
		# Cut down redument [1, 1] and take column data of first and second scores exams
		for x1, x2, y in zip(self.x[1:,0], self.x[1:,1], self.y):
			if (y == 1):
				m, c = '+', 'k'
			else:
				m, c = 'o', 'y'
			plt.plot(x1, x2, m, color = c, mew=2)

		plt.show()

def main():

	path = "/home/rishchen/Source/ML/CourseTF/Tensorflow-Bootcamp-master/Source/LogisiticRegression/ex2data1.txt"
	LR = LogRegression(path)
	#print(LR.sygmoid(0))
	LR.costFunction()
	#print(LR.x[1:,0]
	#print(LR._df[0])
	# LR.plotData()
	#print(LR.sygmoid())
	# print(LR.gradient())

#
if __name__ == '__main__':
	main()