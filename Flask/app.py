import numpy as np
import os
from flask import Flask, request, render_template
from tensorflow import keras
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.utils import load_img,img_to_array
model=load_model(r"wcv.h5")

app=Flask(__name__)



@app.route('/')
def index():
    return render_template('index.html')


@app.route('/details')
def details():
    return render_template('details.html')

@app.route('/type')
def type():
    return render_template('type.html')


@app.route('/home')
def home():
    return render_template("index.html")

@app.route('/input')
def input1():
    return render_template("input.html")
    

@app.route('/predict', methods=["GET","POST"])
def res():
    if request.method=="POST":
        f=request.files['image']
        basepath=os.path.dirname(__file__)
        filepath=os.path.join(basepath,'uploads',f.filename)
        f.save(filepath)
        
        img=load_img(filepath,target_size=(180,180,3))
        x=img_to_array(img)
        x=np.expand_dims(x,axis=0)
        
        img_data=preprocess_input(x)
        prediction=np.argmax(model.predict(img_data), axis=1)
        
        index=['Cloudy','Foggy','Rainy','Shine','Sunrise']
        
        result=str(index[prediction[0]])
        print(result)
        return render_template('output.html', prediction=result)
if __name__ == "__main__":
    app.run(debug=False)