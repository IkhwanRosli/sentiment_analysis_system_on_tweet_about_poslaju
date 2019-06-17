import sys 
import pandas as pd 
from PyQt5 import uic, QtWidgets
from PandasModel2 import PandasModel
from keras.models import model_from_yaml
import numpy as np
import json
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from matplotlib.backends.backend_qt5agg import (NavigationToolbar2QT as NavigationToolbar)
qtCreator = "test.ui"

Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreator)

class MyApp(QtWidgets.QMainWindow, Ui_MainWindow):
	
	def __init__(self):
		QtWidgets.QMainWindow.__init__(self)
		Ui_MainWindow.__init__(self)
		self.setupUi(self)
		
		self.tableView.setSortingEnabled(True)
		self.btnLoadCSV.clicked.connect(self.openFile)
		self.btnLoadCSV.clicked.connect(self.preprocess)
		self.processBtn.clicked.connect(self.processData)
		self.processBtn.clicked.connect(self.createGraph)
		
	def openFile(self):
		global data
		file, _ = QtWidgets.QFileDialog.getOpenFileName(self,'Open File')
		data = pd.read_csv(file,sep = ';')
		if len(data.columns) != 3:
			buildPop()
			exit()
		else:
			dataShow = data
			model = PandasModel(dataShow)
			self.tableView.setModel(model)
			self.tableView.resizeColumnsToContents()
		
	def preprocess(self):
		global review_int
		#Stemming & Remove Stopwords
		data2 = data
		wn = nltk.wordnet.WordNetLemmatizer()
		lc = nltk.stem.SnowballStemmer('english')

		sw = set(stopwords.words('english'))
		hasStop = data2['text'].tolist()
		noStop = []
		for item in hasStop:
			filtered = []
			wt = word_tokenize(item)
			for wo in wt:
				if wo == "not":
					filtered.append(wo)
				elif not wo in sw:
					filtered.append(wo)
			filtered = [wn.lemmatize(w) for w in filtered]
			filtered = [lc.stem(w) for w in filtered]
			noStop.append(' '.join(filtered))
		temp = pd.Series(noStop)
		data2['text'] = temp.values

		#Embedding Word
		with open('vocab.json','r') as json_data:
			voc = json.load(json_data)
			
		from keras.preprocessing.text import one_hot
		import random
		dataList = data2['text'].tolist()
		vocab_int = voc
		data3 =[]
		vocab_size = 200
		for item in dataList:
			notDone = True
			temp1 = (one_hot(item,vocab_size))
			temp2 = item.split()
			for i in range(len(temp2)):
				if temp2[i] in vocab_int:
					continue
				else:
					while notDone:
						if temp1[i] in vocab_int.values():
							temp1[i] = random.randrange(1, vocab_size)
						else:
							notDone = False
				vocab_int[temp2[i]] = temp1[i]
			data3.append(temp1)
		review_int = data3
			
	def buildPop(self):
		name = "Wrong Data Format !!!"
		self.exPopup = examplePopup(name)
		self.exPopup.setGeometry(100, 200, 100, 100)
		self.exPopup.show()
		
	def processData(self):
		import numpy as np
		#Padding        
		seq = 50
		max_words = seq
		features = np.zeros((len(review_int), seq), dtype=int)
		for i,row in enumerate(review_int):
			features[i,-len(row):] = np.array(row)[:seq]

		#Prediction
		import numpy as np
		from keras.models import model_from_yaml

		yaml_file = open('model.yaml', 'r')
		loaded_model_yaml = yaml_file.read()
		yaml_file.close()
		model = model_from_yaml(loaded_model_yaml)
		# load weights into new model
		model.load_weights("model.h5")

		pred = model.predict(features)
		predict = (pred > 0.5)

		import itertools
		prediction = list(itertools.chain(*predict))
		prediction = [item.astype(str) for item in prediction]
		
		data["label"] = prediction
		
	def createGraph(self):
		colors = ['lime','yellow']
		tempPos = (data["label"] == "True").sum()
		tempNeg = (data["label"] == "False").sum()
	
		lblReview = ["Positive","Negative"]
		review = [tempPos,tempNeg]
		
		self.graph1.canvas.axes.pie(review, labels=lblReview, autopct='%1.1f%%',colors = colors)
		self.graph1.canvas.axes.axis('equal')
		self.graph1.canvas.axes.set_visible(True)
		self.graph1.canvas.draw()
		
		countPos1 = 0
		countNeg1 = 0
		countPos2 = 0
		countNeg2 = 0
		countPos3 = 0
		countNeg3 = 0
		
		for i,row in data.iterrows():
			if row["category"] == 1:
				if row["label"] == "True":
					countPos1 += 1
				else:
					countNeg1 += 1
					
			elif row["category"] == 2:
				if row["label"] == "True":
					countPos2 += 1
				else:
					countNeg2 += 1
					
			elif row["category"] == 3:
				if row["label"] == "True":
					countPos3 += 1
				else:
					countNeg3 += 1

		self.graph2.canvas.axes.pie([countPos1,countNeg1], labels=lblReview, autopct='%1.1f%%',colors = colors)
		self.graph2.canvas.axes.axis('equal')
		self.graph2.canvas.axes.set_visible(True)
		self.graph2.canvas.draw()
		
		self.graph3.canvas.axes.pie([countPos2,countNeg2], labels=lblReview, autopct='%1.1f%%',colors = colors)
		self.graph3.canvas.axes.axis('equal')
		self.graph3.canvas.axes.set_visible(True)
		self.graph3.canvas.draw()		
		
		self.graph4.canvas.axes.pie([countPos3,countNeg3], labels=lblReview, autopct='%1.1f%%',colors = colors)
		self.graph4.canvas.axes.axis('equal')
		self.graph4.canvas.axes.set_visible(True)
		self.graph4.canvas.draw()		

if __name__ == '__main__':
	app = QtWidgets.QApplication(sys.argv)
	window = MyApp()
	window.show()
	sys.exit(app.exec_())