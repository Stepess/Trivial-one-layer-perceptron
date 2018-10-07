import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
 
def setSamplesOnPlot(frame):
	for i in range(len(frame)):
		if(frame.iloc[i][2] == 1.0):
			plt.scatter(frame.iloc[i][0],frame.iloc[i][1],c='black',marker='o')
		else:
			plt.scatter(frame.iloc[i][0],frame.iloc[i][1],c='blue',marker='o')
	
#learning process of neuron
def learn(x_train, y_train):
	y = []
	w = [0,0,0]
	for i in range(80):
		x = []
		x.append(1) #bias neuron
		for j in x_train.values.tolist()[i]:
			x.append(j)
		w = update(w,x,actiation(w,x),y_train.values.tolist()[i][0])
	return w

def actiation(w,x):
    theta = 0.5
    res = np.dot(w,x)
    if res >=  theta:
        return 1
    else:
        return 0

def update(w,x,expectedOutput,actualOutput):
    #new_w = [0,0,0]
    for k in range(len(w)):
        w[k] = w[k] + 0.1*(actualOutput-expectedOutput)*x[k]
    return w
	
#check study accuracy
def test(w, x_test, y_test):
	y=[]
	for i in range(20):
		x = []
		x.append(1)
		for j in x_test.values.tolist()[i]:
			x.append(j)
		y.append(actiation(w,x))
	y_real = y_test.values.tolist()   
	count = 0
	for i in range(len(y)):
		if y[i] == y_real[i][0]:
			count+=1
	return (count*1.0/len(y))

		
def buildFunctionOfSeparation(w):
	f = []
	for x in np.arange(0,1,0.1):
		f.append(actiation(w,[1,x,x])) #1 - for bias neuron
	plt.plot(np.arange(0,1,0.1),f, color='red')
	

if __name__ == "__main__":
	frame = pd.read_csv('data03.csv', sep=';')
	frame = frame.sample(frac=1) #shafle dataframe in random order

	train_data = frame[:80]
	test_data = frame[-20:]

	x_train = train_data.iloc[:,0:2] # first two columns of data frame with all rows
	y_train = train_data.iloc[:,2:3] # 2nd column of data frame with all rows

	x_test = test_data.iloc[:,0:2] 
	y_test = test_data.iloc[:,2:3]
	
	setSamplesOnPlot(frame)
	plt.show()
	
	w = learn(x_train, y_train)
	accuracy = test(w, x_test, y_test)*100
	print("Your precision is {}% ".format(accuracy))
	print(w)
	
	setSamplesOnPlot(frame)
	buildFunctionOfSeparation(w)
	plt.show()
	
	