from flask import Flask,request,render_template,redirect,url_for
import numpy as np
import tensorflow as tf
from PIL import Image
app=Flask(__name__)
import os
import secrets
keras=tf.keras


@app.route('/')
def home():
	test_model = tf.keras.models.load_model('training7.h5')
	test_gen = tf.keras.preprocessing.image.ImageDataGenerator(
								preprocessing_function=keras.applications.mobilenet.preprocess_input).flow_from_directory(directory='testing',
                                                                                            								target_size=(224, 224),
                                                                                                    						batch_size=1,
                                                                                            								shuffle=False)
	test_gen.reset()
	p = test_model.predict_generator(test_gen, steps=1)
	predicted_label = np.argmax(p, axis=1)
	if(predicted_label[0]==1):
		return "pothole"
	else:
		return "not-pothole"

def save_and_upload(file):
    random_hex = secrets.token_hex(8)
    _, f_ext = os.path.splitext(file.filename)
    picture_fn = random_hex + f_ext
    picture_path = os.path.join(app.root_path, 'testing/All_classes', picture_fn)
    output_size = (224, 224)
    i = Image.open(file)
    i.thumbnail(output_size)
    i.save(picture_path)
    return picture_fn

@app.route('/account',methods=['GET','POST'])
def account():
	if request.method == 'POST':
		file=request.files.get('file')
		print(file.filename)
		save_and_upload(file)
		return redirect(url_for('account'))
	return render_template('account.html')


if(__name__=='__main__'):
    app.run(port=5005,debug=True)
