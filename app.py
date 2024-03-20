from flask import Flask, render_template, request
import  sklearn
import pickle
from tensorflow.keras.models import model_from_json
import tensorflow as tf
import numpy as np
import cv2


l1 = ['itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 'shivering', 'chills', 'joint_pain',
      'stomach_pain', 'acidity', 'ulcers_on_tongue', 'muscle_wasting', 'vomiting', 'burning_micturition', 'spotting_urination', 'fatigue',
      'weight_gain', 'anxiety', 'mood_swings', 'weight_loss', 'restlessness', 'lethargy', 'patches_in_throat',
      'irregular_sugar_level', 'cough', 'high_fever', 'sunken_eyes', 'breathlessness', 'sweating', 'dehydration', 'indigestion',
      'headache', 'yellowish_skin', 'dark_urine', 'nausea', 'loss_of_appetite', 'pain_behind_the_eyes', 'back_pain', 'constipation',
      'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine', 'yellowing_of_eyes', 'acute_liver_failure', 'fluid_overload',
      'swelling_of_stomach', 'swelled_lymph_nodes', 'malaise', 'blurred_and_distorted_vision', 'phlegm', 'throat_irritation',
      'redness_of_eyes', 'sinus_pressure', 'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs', 'fast_heart_rate',
      'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool', 'irritation_in_anus', 'neck_pain', 'dizziness', 'cramps',
      'bruising', 'obesity', 'swollen_legs', 'swollen_blood_vessels', 'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails',
      'swollen_extremities', 'excessive_hunger', 'extra_marital_contacts', 'drying_and_tingling_lips', 'slurred_speech', 'knee_pain', 'hip_joint_pain',
      'muscle_weakness', 'stiff_neck', 'swelling_joints', 'movement_stiffness', 'spinning_movements', 'loss_of_balance', 'unsteadiness', 'weakness_of_one_body_side',
      'loss_of_smell', 'bladder_discomfort', 'foul_smell_of urine', 'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching', 'toxic_look_(typhus)',
      'depression', 'irritability', 'muscle_pain', 'altered_sensorium', 'red_spots_over_body', 'belly_pain', 'abnormal_menstruation',
      'watering_from_eyes', 'increased_appetite', 'polyuria', 'family_history', 'mucoid_sputum', 'rusty_sputum', 'lack_of_concentration', 'visual_disturbances',
      'receiving_blood_transfusion', 'receiving_unsterile_injections', 'coma', 'stomach_bleeding', 'distention_of_abdomen', 'history_of_alcohol_consumption',
      'fluid_overload', 'blood_in_sputum', 'prominent_veins_on_calf', 'palpitations', 'painful_walking', 'pus_filled_pimples', 'blackheads', 'scurrying', 'skin_peeling',
      'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails', 'blister', 'red_sore_around_nose', 'yellow_crust_ooze']
disease = ['Fungal infection', 'Allergy', 'GERD', 'Chronic cholestasis', 'Drug Reaction',
           'Peptic ulcer diseae', 'AIDS', 'Diabetes', 'Gastroenteritis', 'Bronchial Asthma', 'Hypertension',
           ' Migraine', 'Cervical spondylosis',
           'Paralysis (brain hemorrhage)', 'Jaundice', 'Malaria', 'Chicken pox', 'Dengue', 'Typhoid', 'hepatitis A',
           'Hepatitis B', 'Hepatitis C', 'Hepatitis D', 'Hepatitis E', 'Alcoholic hepatitis', 'Tuberculosis',
           'Common Cold', 'Pneumonia', 'Dimorphic hemmorhoids(piles)',
           'Heartattack', 'Varicoseveins', 'Hypothyroidism', 'Hyperthyroidism', 'Hypoglycemia', 'Osteoarthristis',
           'Arthritis', '(vertigo) Paroymsal  Positional Vertigo', 'Acne', 'Urinary tract infection', 'Psoriasis',
           'Impetigo']




heart = open("models/HeartDisease.json", "r")
loaded_data = heart.read()
heart_model = model_from_json(loaded_data)


diabetes = open("models/Diabetes.json", "r")
loaded_data = diabetes.read()
diabetes_model = model_from_json(loaded_data)


stroke = open("models/Stroke.json", "r")
loaded_data = stroke.read()
stroke_model = model_from_json(loaded_data)


kidney = open("models/KidneyDisease.json", "r")
loaded_data = kidney.read()
kidney_model = model_from_json(loaded_data)


liver = open("models/LiverDisease.json", "r")
loaded_data = liver.read()
liver_model = model_from_json(loaded_data)

symptoms_model = pickle.load(open('models/Symptom', 'rb'))

app = Flask(__name__, static_url_path='/static')



@app.route('/')
def login():
    return render_template('login.html')


@app.route('/home')
def home():
    return render_template('Home.html')


@app.route('/about')
def about():
    return render_template('About.html')


@app.route('/contact')
def contact():
    return render_template('Contact.html')


@app.route('/heart')
def heart():
    return render_template('Heart.html')


@app.route('/diabetes')
def diabetes():
    return render_template('Diabetes.html')


@app.route('/kidney')
def kidney():
    return render_template('Kidney.html')


@app.route('/liver')
def liver():
    return render_template('Liver.html')


@app.route('/stroke')
def stroke():
    return render_template('stroke.html')


@app.route('/symptoms')
def symptoms():
    return render_template('Symptoms.html')

@app.route('/logins')
def logins():
    return render_template('login.html')


@app.route('/result_heart',  methods=['GET', 'POST'])
def result_heart():
    if request.method == 'POST':
        age = float(request.form['age'])
        gender = float(request.form['gender'])
        chestPainType = float(request.form['chestPainType'])
        restingBP = float(request.form['restingBP'])
        cholesterol = float(request.form['cholesterol'])
        fastingBS = float(request.form['fastingBS'])
        restingECG = float(request.form['restingECG'])
        maxHR = float(request.form['maxHR'])
        exerciseAngina = float(request.form['exerciseAngina'])
        oldpeak = float(request.form['oldpeak'])
        st_slope = float(request.form['st_slope'])
        lst = [age, gender, chestPainType, restingBP, cholesterol,
               fastingBS, restingECG, maxHR, exerciseAngina, oldpeak, st_slope]
        arr = [np.array([lst, ])]
        output = heart_model.predict(arr)
        if output[0][0] == 1:
            output *= 99
        else:
            output *= 100

        return render_template('Result_heart.html', prediction_text="{:.2f} %".format(output[0][0]))


@app.route('/result_diabetes',  methods=['GET', 'POST'])
def result_diabetes():
    if request.method == 'POST':
        pregnancies = float(request.form.get('pregnancies'))
        glucose = float(request.form.get('glucose'))
        blood_pressure = float(request.form.get('bloodPressure'))
        skin_thickness = float(request.form.get('skinthickness'))
        insulin = float(request.form.get('insulin'))
        bmi = float(request.form.get('bmi'))
        diabetes_pedigree_function = float(
            request.form.get('diabetespedigreefunction'))
        age = float(request.form.get('age'))
        lst = [pregnancies, glucose, blood_pressure,
               skin_thickness, insulin, bmi, diabetes_pedigree_function, age]
        arr = [np.array([lst, ])]
        output = diabetes_model.predict(arr)
        if output[0][0] == 1:
            output *= 99
        else:
            output *= 100

    return render_template('Result_diabetes.html', prediction_text="{:.2f} %".format(output[0][0]))


@app.route('/result_stroke',  methods=['GET', 'POST'])
def result_stroke():
    if request.method == 'POST':
        gender = request.form.get('gender', type=int)
        age = request.form.get('age', type=int)
        hypertension = request.form.get('hypertension', type=int)
        heart_disease = request.form.get('heart_disease', type=int)
        ever_married = request.form.get('ever_married', type=int)
        work_type = request.form.get('work_type', type=int)
        Residence_type = request.form.get('Residence_type', type=int)
        avg_glucose_level = request.form.get('avg_glucose_level', type=float)
        bmi = request.form.get('bmi', type=float)
        smoking_status = request.form.get('smoking_status', type=int)
        lst = [gender, age, hypertension, heart_disease, ever_married, work_type, Residence_type, avg_glucose_level, bmi, smoking_status]
        arr = [np.array([lst, ])]
        output = stroke_model.predict(arr)
        if output[0][0] == 1:
            output *= 99
        else:
            output *= 100

    return render_template('Result_stroke.html', prediction_text="{:.2f} %".format(output[0][0]))



@app.route('/result_kidney',  methods=['GET', 'POST'])
def result_kidney():
    if request.method == 'POST':
        lst = [float(x) for x in request.form.values()]
        arr = [np.array([lst, ])]
        output = kidney_model.predict(arr)
        if output[0][0] == 1:
            output *= 99
        else:
            output *= 100

    return render_template('Result_kidney.html', prediction_text="{:.2f} %".format(output[0][0]))


@app.route('/result_liver',  methods=['GET', 'POST'])
def result_liver():
    if request.method == 'POST':
        lst = [float(x) for x in request.form.values()]
        arr = [np.array([lst, ])]
        output = liver_model.predict(arr)
        if output[0][0] == 1:
            output *= 99
        else:
            output *= 100

    return render_template('Result_liver.html', prediction_text="{:.2f} %".format(output[0][0]))



@app.route('/result_symptoms',  methods=['GET', 'POST'])
def result_symptoms():
    if request.method == 'POST':
        lst = [x for x in request.form.values()]
        arr = np.zeros(132)
        for k in lst:
            for i, j in enumerate(l1):
                if k == j:
                    arr[i] = 1
        output = symptoms_model.predict([arr])

    return render_template('Result_symptoms.html', prediction_text="{}".format(disease[output[0]]))


if __name__ == '__main__':

    app.run(port=8080)
