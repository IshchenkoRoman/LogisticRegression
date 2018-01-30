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

import scipy.optimize as opt
import os

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
		self.x = np.c_[ones, x_1, x_2]
		self.y = self._df.iloc[:,2]

	def sygmoid(self, data = None):

		if (data is None):
			data = self.x
		g_z = 1 / (1 + np.exp(-data))
		return (g_z)

	def costFunction(self, thetta, X, y):
		
		# thetta = np.zeros((3, 1))
		# ones = np.ones((self._l))
		hypotesis = self.sygmoid(np.dot(X, thetta))
		log_hypotesis = np.log(hypotesis)
		log_hypotesis_2 = np.log(np.subtract(1, hypotesis)) # WTF? Here I swap "ones on "1" and all good
		first_part = np.dot(-y, log_hypotesis)
		second_part = np.dot((np.subtract(1, y)), log_hypotesis_2)

		list_ = np.subtract(first_part, second_part)
		J_cost = (1 / self._l) * np.sum(list_)
		return (J_cost)
		
	def gradient(self, thetta, X, y):

		hypotesis = self.sygmoid(np.dot(X, thetta))
		hypotesis = np.reshape(hypotesis, self._l)
		list_ = np.dot(np.subtract(hypotesis, y), X)

		return (np.dot(1 / self._l, list_))

	# def costFunction(self, thetta=np.zeros((3, 1))):
		
	# 	# thetta = np.zeros((3, 1))
	# 	# ones = np.ones((self._l))
	# 	hypotesis = self.sygmoid(np.dot(self.x, thetta))
	# 	log_hypotesis = np.log(hypotesis)
	# 	log_hypotesis_2 = np.log(np.subtract(1, hypotesis)) # WTF? Here I swap "ones on "1" and all good
	# 	first_part = np.dot(-self.y, log_hypotesis)
	# 	second_part = np.dot((np.subtract(1, self.y)), log_hypotesis_2)

	# 	list_ = np.subtract(first_part, second_part)
	# 	J_cost = (1 / self._l) * np.sum(list_)
	# 	return (J_cost)
		
	# def gradient(self, thetta=np.zeros((3, 1))):

	# 	hypotesis = self.sygmoid(np.dot(self.x, thetta))
	# 	hypotesis = np.reshape(hypotesis, self._l)
	# 	list_ = np.dot(np.subtract(hypotesis, self.y), self.x)
	# 	sum_ = np.sum(list_)

	# 	return ((1 / self._l) * sum_)

	def optimizeFunc(self, thetta=np.zeros(3)):
			
			result = opt.fmin_tnc(func=self.costFunction, x0=thetta, fprime=self.gradient, args=(self.x, self.y))
			cs = self.costFunction(result[0], self.x, self.y)
			return (result[0])

	def predict(self, thetta=np.zeros(3), exam1=45, exam2=85):

		arr = np.ones(3)
		arr[1] = exam1
		arr[2] = exam2

		dot = np.dot(arr, thetta)
		res = self.sygmoid(dot)
		print(res)
		return (1 if res > 0.5 else 0)

	def _plot(self, label_x, label_y, label_pos, label_neg, axes=None):

		neg = self._df.iloc[:,2] == 0
		pos = self._df.iloc[:,2] == 1

		if axes == None:
			axes = plt.gca()

		axes.scatter(self._df[pos][0], self._df[pos][1], marker='+', c='k', s=60, linewidths=2, label=label_pos)
		axes.scatter(self._df[neg][0], self._df[neg][1], c='y', s=60, label=label_neg)
		axes.set_xlabel(label_x)
		axes.set_ylabel(label_y)
		axes.legend(frameon = True, fancybox = True)


	def plotDecisionBoundary(self, label_x, label_y, label_pos, label_neg, thetta=np.ones(3), exam1=45, exam2=85, axes=None, type=1):

		if type == 2:
			str_label_input_ex = "({0}, {1})".format(exam1, exam2)
			plt.scatter(exam1, exam2, s=60, c='r', marker='v', label=str_label_input_ex)
			x1_min, x1_max = self.x[:,1].min(), self.x[:,1].max()
			x2_min, x2_max = self.x[:,2].min(), self.x[:,2].max()
			xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
			h = self.sygmoid(np.c_[np.ones((xx1.ravel().shape[0], 1)), xx1.ravel(), xx2.ravel()].dot(thetta))
			h = h.reshape(xx1.shape)
			plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors='b')

		self._plot(label_x, label_y, label_pos, label_neg, axes)

		plt.show()


def main():

	#path = "/home/rishchen/Source/ML/CourseTF/Tensorflow-Bootcamp-master/Source/LogisiticRegression/ex2data1.txt"
	path = os.getcwd() + "/ex2data1.txt"
	LR = LogRegression(path)
	print(LR.sygmoid(0))
	z = np.zeros(3)
	# print(LR.costFunction(z))
	thetta = LR.optimizeFunc()
	print("thetta = {0}".format(thetta))
	LR.predict(thetta)
	#print(LR.x[1:,0]
	#print(LR._df[0])
	#LR.plotData1()
	LR.plotDecisionBoundary('Exam score 1', 'Exam score 2', 'Admitted', 'Not admitted', thetta=thetta, type=2)
	#print(LR.sygmoid())
	# print(LR.gradient())

if __name__ == '__main__':
	main()