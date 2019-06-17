import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.cross_validation import train_test_split
from sklearn.svm import 

df = pd.read_csv("data.csv", sep = ";")

y = df.label
x = df['text'].tolist()

onehot_enc = MultiLabelBinarizer()
onehot_enc.fit(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=40)

model = LinearSVC()
model.fit(onehot_enc.transform(x_train), y_train)
score = model.score(onehot_enc.transform(x_test), y_test)
print("THE accuracy of the model is {}%".format(score*100.0))