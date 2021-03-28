#!/usr/bin/env python
# coding: utf-8

# In[2]:


from flask import Flask, request
import pickle
import pandas as pd

# In[3]:
import flask
app = Flask(__name__)

# In[5]:
@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/template')
def index():
    return flask.render_template('santander_index.html')


with open('final_model.pkl','rb') as file:
        model=pickle.load(file)
@app.route('/predict', methods=['POST'])

def predict():
    file = request.files['file']
    data=pd.read_csv(file)
    idcode=data['ID_code']
    columns=data.drop(['ID_code'],axis=1).columns.values
    
    data['max_val']=data[columns].max(axis=1)
    data['std_dev']=data[columns].std(axis=1)
    data['skew']=data[columns].skew(axis=1)
    data['kurtosis']=data[columns].kurtosis(axis=1)
    data['sum']=data[columns].sum(axis=1)
    x_test=data.drop(['ID_code'],axis=1)
    
    result=pd.DataFrame()
    result['id_code']=idcode
    result['Will make transaction']= model.predict(x_test)
    for i in range(len(result['Will make transaction'])):
        if result['Will make transaction'][i]==0:
            result['Will make transaction'][i]='no'
        else:
            result['Will make transaction'][i]='yes'        
            
    return result.to_json(orient='records',lines=True)
#jsonify(result.to_dict())            
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080,debug=True)    

# In[ ]:




