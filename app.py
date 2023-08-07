from flask import Flask, render_template, request
# from imblearn.over_sampling import SMOTE
import joblib
import os
import numpy as np
import pickle
import tensorflow as tf
import keras



app = Flask(__name__, template_folder='templates', static_folder='static')


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/result", methods=['POST', 'GET'])
def result():
    gender = int(request.form['gender'])
    age = int(request.form['age'])
    hypertension = int(request.form['hypertension'])
    heart_disease = int(request.form['heart_disease'])
    ever_married = int(request.form['ever_married'])
    govt_job = int(request.form['work_type'])
    residence_type = int(request.form['Residence_type'])
    avg_glucose_level = float(request.form['avg_glucose_level'])
    bmi = float(request.form['bmi'])
    formerly = int(request.form['smoke_status'])


    x = np.array([gender, age, hypertension, heart_disease, ever_married, govt_job, residence_type,
                  avg_glucose_level, bmi, formerly]).reshape(1, -1)

    scaler_path = os.path.join('C:/Users/Prason/Desktop/capstone', 'models/scaler.pkl')
    with open(scaler_path, 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)

    print(x)
    x = scaler.transform(x)
    print(x)

    model_path = os.path.join('C:/Users/Prason/Desktop/capstone', 'models/ntlrmodel.sav')
    ntlrmodel = joblib.load(model_path)

    model_path_knn = os.path.join('C:/Users/Prason/Desktop/capstone/', 'models/knnmodel.sav')
    knnmodel = joblib.load(model_path_knn)

    model_path_dect = os.path.join('C:/Users/Prason/Desktop/capstone/', 'models/dectmodel.sav')
    dectmodel = joblib.load(model_path_dect)

    model_path_lsvc = os.path.join('C:/Users/Prason/Desktop/capstone/', 'models/lsvcmodel.sav')
    lsvcmodel = joblib.load(model_path_lsvc)

    model_path_svc = os.path.join('C:/Users/Prason/Desktop/capstone/', 'models/svcmodel.sav')
    svcmodel = joblib.load(model_path_svc)

    model_path_rfc = os.path.join('C:/Users/Prason/Desktop/capstone/', 'models/rfcmodel.sav')
    rfcmodel = joblib.load(model_path_rfc)

    model_path_cnna = os.path.join('C:/Users/Prason/Desktop/capstone', 'models/cnnamodel.sav')
    cnnamodel = joblib.load(model_path_cnna)

    model_path_cnnb = tf.keras.models.load_model('./cnnb/modeldd')
    # cnnbmodel = joblib.load(model_path_cnnb)

    model_path_lstm = tf.keras.models.load_model('./lstm/modellstm')
    # model_Conv1D = tf.keras.models.load_model('./cnnb/modeldd')
    # Y_predConv1D = model_Conv1D.predict(x)

    Y_pred_ntlr = ntlrmodel.predict(x)
    Y_pred_knn = knnmodel.predict(x)
    Y_pred_dect = dectmodel.predict(x)
    Y_pred_lsvc = lsvcmodel.predict(x)
    Y_pred_svc = svcmodel.predict(x)
    Y_pred_rfc = rfcmodel.predict(x)
    Y_pred_cnna = cnnamodel.predict(x)
    Y_pred_cnnb = model_path_cnnb.predict(x)
    # new_x = tf.expand_dims(x, axis=1)
    # new_x_sq = tf.squeeze
    Y_pred_lstm = model_path_lstm.predict(tf.expand_dims(x, axis=1))

    print(Y_pred_ntlr, Y_pred_knn, Y_pred_dect, Y_pred_lsvc, Y_pred_svc, Y_pred_rfc, Y_pred_cnna, Y_pred_cnnb, Y_pred_lstm)
    finalvote = (Y_pred_ntlr + Y_pred_knn + Y_pred_dect + Y_pred_lsvc + Y_pred_svc + Y_pred_rfc + Y_pred_cnna +Y_pred_cnnb + Y_pred_lstm) / 9
    print(finalvote)

    # for No Stroke Risk
    if finalvote < 0.5:
        return render_template('nonstroke.html')
    else:
        return render_template('stroke.html')



if __name__ == "__main__":
    app.run(debug=True, port=7384)