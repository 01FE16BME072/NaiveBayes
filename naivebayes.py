import numpy as np
from matplotlib import pyplot as plt
from matplotlib import style
import warnings

style.use('ggplot')

datasets = {'green':[[1,2],[2,3],[3,1]],'red':[[4,5],[6,7],[7,6]]}
newFeature = [3.5,3.5]

# for i in datasets:
# 	for j in datasets[i]:
# 		plt.scatter(j[0],j[1],s = 50,color = i)
# plt.scatter(newFeature[0],newFeature[1],s = 50,color = 'yellow')
# plt.show()

class Naivebayes:
	def __init__(self,data,predict,k = 3):
		self.data = data
		self.k = k
		self.predict = predict

		if len(data) >= k:
			warnings.warn('Dumbass!!!!')

		distances = []
		distances_grp = []
		for group in data:
			for features in data[group]:
				euclid = np.linalg.norm(np.array(features)-np.array(predict))
				distances_grp.append([euclid,group])
				distances.append(euclid)
		#print(distances)
		count = 0
		for i in distances:
			if i <= k:
				count = count + 1
		print(count)

		self.votes = [i[0] for i in sorted(distances_grp)[:count]]
		self.votes_group = [i[1] for i in sorted(distances_grp)[:count]]
		#print(votes)
		#print(votes_group)
		self.a = float(len(self.votes))
		self.b = len(self.votes_group)
		#print(a,b)

	def fit(self,data):
		self.data = data
		Observations_green = []
		Observations_red = []

		for i in data:
			#print i
			if i == 'green':
				for j in data[i]:
					Observations_green.append(j)
			else:
				for j in data[i]:
					Observations_red.append(j)

		# print(Observations_green)
		# print(Observations_red)
		# print(len(Observations_green))
		# print(len(Observations_red))
		
		green_points = float(len(Observations_green))
		red_points = float(len(Observations_red))

		totalpoints = float(green_points + red_points)
		#print(totalpoints)
		#print(self.a)
		#print(self.votes_group)
		countg = 0.0
		countr = 0.0
		for i in self.votes_group:
			if i == 'green':
				countg = countg +1
			else:
				countr = countr +1
		#print countg
		#print countr
		# Probability_green = 0.0
		Probability_green = float(green_points/totalpoints)
		#print(Probability_green)
		Probability_red = float(red_points/totalpoints)
		#print(Probability_red)
		Probability_feature = float(self.a/totalpoints)
		#print(Probability_feature)
		Probability_feature_green = float(countg/green_points)
		#print(Probability_feature_green)
		Probability_feature_red = float(countr/red_points)
		#print(Probability_feature_red)

		Probability_green_feature = float((Probability_feature_green*Probability_green)/Probability_feature)
		Probability_red_feature = float((Probability_feature_red*Probability_red)/Probability_feature)

		#print(Probability_green_feature)
		if Probability_green_feature > Probability_red_feature:
			print 'The given new datapoint belongs to GREEN class with a Probability of:', Probability_green_feature,'for a', self.k, 'unit radius'
		else:
			print 'The given new datapoint belongs to RED class with a Probability of:', Probability_red_feature,'for a', self.k, 'unit radius'
	
	def visualize(self):
		fig = plt.figure()
		ax = fig.add_subplot(1,1,1)
		a = plt.Circle((self.predict[0],self.predict[1]),radius = self.k,fill = False,color = 'blue')
		for i in self.data:
			for j in self.data[i]:
				plt.scatter(j[0],j[1],s = 50,color = i)
		plt.scatter(self.predict[0],self.predict[1],s = 50,color = 'yellow')
		ax.add_patch(a)
		plt.show()

classifier = Naivebayes(datasets,newFeature,k = 3)
classifier.fit(datasets)
classifier.visualize()