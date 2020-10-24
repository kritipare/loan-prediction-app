from flask import Flask,render_template,request
import pickle
import numpy as np 
from loan_predict2 import InterestRate
app = Flask(__name__)


model = pickle.load(open('model.pkl','rb'))
@app.route('/')
# def hello_world():
# 	return 'hello world'
def hello_world():
    return render_template("index2.html")



@app.route('/predict',methods=['POST','GET'])
def predict():
	int_features=[int(x) for x in request.form.values()]
	final=[np.array(int_features)]

	ans = InterestRate(int_features[1],int_features[0])
	ans = int(ans[0])
	return render_template('index2.html',pred=ans)

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=4000,debug=True)
