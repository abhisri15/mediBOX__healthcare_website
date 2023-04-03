from flask import Flask, request, jsonify,render_template
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

# Load the Random Forest Classifier model
model = pickle.load(open("model_disease.pkl", "rb"))

# Define the route for the home page
@app.route("/", methods = ["GET","POST"])
def home():
    return render_template('Input_1.html')

# Define the route for the prediction page
@app.route("/diseasePredictor", methods=["GET","POST"])
def predict():
    #render_template('Input.html')
    # Get the symptoms from the request form
    # symptoms = request.form.to_dict()
    # Handle the form submission
    if request.method == 'POST':
        # Get the list of selected features
        # selected_features = request.form.getlist('weight_loss')
        features = []
        feature = request.form.get('itching')
        features.append(feature)

        feature = request.form.get('skin_rash')
        features.append(feature)

        feature = request.form.get('nodal_skin_eruptions')
        features.append(feature)

        feature = request.form.get('continuous_sneezing')
        features.append(feature)

        feature = request.form.get('shivering')
        features.append(feature)

        feature = request.form.get('chills')
        features.append(feature)

        feature = request.form.get('joint_pain')
        features.append(feature)

        feature = request.form.get('stomach_pain')
        features.append(feature)

        feature = request.form.get('acidity')
        features.append(feature)

        feature = request.form.get('ulcers_on_tongue')
        features.append(feature)

        feature = request.form.get('muscle_wasting')
        features.append(feature)

        feature = request.form.get('vomiting')
        features.append(feature)

        feature = request.form.get('burning_micturition')
        features.append(feature)

        feature = request.form.get('spotting_ urination')
        features.append(feature)

        feature = request.form.get('fatigue')
        features.append(feature)

        feature = request.form.get('weight_gain')
        features.append(feature)

        feature = request.form.get('anxiety')
        features.append(feature)

        feature = request.form.get('cold_hands_and_feets')
        features.append(feature)

        feature = request.form.get('mood_swings')
        features.append(feature)

        feature = request.form.get('weight_loss')
        features.append(feature)

        feature = request.form.get('restlessness')
        features.append(feature)

        feature = request.form.get('lethargy')
        features.append(feature)

        feature = request.form.get('patches_in_throat')
        features.append(feature)

        feature = request.form.get('irregular_sugar_level')
        features.append(feature)

        feature = request.form.get('cough')
        features.append(feature)

        feature = request.form.get('high_fever')
        features.append(feature)

        feature = request.form.get('sunken_eyes')
        features.append(feature)

        feature = request.form.get('breathlessness')
        features.append(feature)

        feature = request.form.get('sweating')
        features.append(feature)

        feature = request.form.get('dehydration')
        features.append(feature)

        feature = request.form.get('indigestion')
        features.append(feature)

        feature = request.form.get('headache')
        features.append(feature)

        feature = request.form.get('yellowish_skin')
        features.append(feature)

        feature = request.form.get('dark_urine')
        features.append(feature)

        feature = request.form.get('nausea')
        features.append(feature)

        feature = request.form.get('loss_of_appetite')
        features.append(feature)

        feature = request.form.get('pain_behind_the_eyes')
        features.append(feature)

        feature = request.form.get('back_pain')
        features.append(feature)

        feature = request.form.get('constipation')
        features.append(feature)

        feature = request.form.get('abdominal_pain')
        features.append(feature)

        feature = request.form.get('diarrhoea')
        features.append(feature)

        feature = request.form.get('mild_fever')
        features.append(feature)

        feature = request.form.get('yellow_urine')
        features.append(feature)

        feature = request.form.get('yellowing_of_eyes')
        features.append(feature)

        feature = request.form.get('acute_liver_failure')
        features.append(feature)

        feature = request.form.get('acute_liver_failure')
        features.append(feature)

        feature = request.form.get('swelling_of_stomach')
        features.append(feature)

        feature = request.form.get('swelled_lymph_nodes')
        features.append(feature)

        feature = request.form.get('malaise')
        features.append(feature)

        feature = request.form.get('blurred_and_distorted_vision')
        features.append(feature)

        feature = request.form.get('phlegm')
        features.append(feature)

        feature = request.form.get('throat_irritation')
        features.append(feature)

        feature = request.form.get('redness_of_eyes')
        features.append(feature)

        feature = request.form.get('sinus_pressure')
        features.append(feature)

        feature = request.form.get('runny_nose')
        features.append(feature)

        feature = request.form.get('congestion')
        features.append(feature)

        feature = request.form.get('chest_pain')
        features.append(feature)

        feature = request.form.get('weakness_in_limbs')
        features.append(feature)

        feature = request.form.get('fast_heart_rate')
        features.append(feature)

        feature = request.form.get('pain_during_bowel_movements')
        features.append(feature)

        feature = request.form.get('pain_in_anal_region')
        features.append(feature)

        feature = request.form.get('bloody_stool')
        features.append(feature)

        feature = request.form.get('irritation_in_anus')
        features.append(feature)

        feature = request.form.get('neck_pain')
        features.append(feature)

        feature = request.form.get('dizziness')
        features.append(feature)

        feature = request.form.get('cramps')
        features.append(feature)

        feature = request.form.get('bruising')
        features.append(feature)

        feature = request.form.get('obesity')
        features.append(feature)

        feature = request.form.get('swollen_legs')
        features.append(feature)

        feature = request.form.get('swollen_blood_vessels')
        features.append(feature)

        feature = request.form.get('puffy_face_and_eyes')
        features.append(feature)

        feature = request.form.get('enlarged_thyroid')
        features.append(feature)

        feature = request.form.get('brittle_nails')
        features.append(feature)

        feature = request.form.get('swollen_extremeties')
        features.append(feature)

        feature = request.form.get('excessive_hunger')
        features.append(feature)

        feature = request.form.get('extra_marital_contacts')
        features.append(feature)

        feature = request.form.get('drying_and_tingling_lips')
        features.append(feature)

        feature = request.form.get('slurred_speech')
        features.append(feature)

        feature = request.form.get('knee_pain')
        features.append(feature)

        feature = request.form.get('hip_joint_pain')
        features.append(feature)

        feature = request.form.get('muscle_weakness')
        features.append(feature)

        feature = request.form.get('stiff_neck')
        features.append(feature)

        feature = request.form.get('swelling_joints')
        features.append(feature)

        feature = request.form.get('movement_stiffness')
        features.append(feature)

        feature = request.form.get('spinning_movements')
        features.append(feature)

        feature = request.form.get('loss_of_balance')
        features.append(feature)

        feature = request.form.get('unsteadiness')
        features.append(feature)

        feature = request.form.get('weakness_of_one_body_side')
        features.append(feature)

        feature = request.form.get('loss_of_smell')
        features.append(feature)

        feature = request.form.get('bladder_discomfort')
        features.append(feature)

        feature = request.form.get('foul_smell_of urine')
        features.append(feature)

        feature = request.form.get('continuous_feel_of_urine')
        features.append(feature)

        feature = request.form.get('passage_of_gases')
        features.append(feature)

        feature = request.form.get('internal_itching')
        features.append(feature)

        feature = request.form.get('toxic_look_(typhos)')
        features.append(feature)

        feature = request.form.get('depression')
        features.append(feature)

        feature = request.form.get('irritability')
        features.append(feature)

        feature = request.form.get('muscle_pain')
        features.append(feature)

        feature = request.form.get('altered_sensorium')
        features.append(feature)

        feature = request.form.get('red_spots_over_body')
        features.append(feature)

        feature = request.form.get('belly_pain')
        features.append(feature)

        feature = request.form.get('abnormal_menstruation')
        features.append(feature)

        feature = request.form.get('dischromic _patches')
        features.append(feature)

        feature = request.form.get('watering_from_eyes')
        features.append(feature)

        feature = request.form.get('increased_appetite')
        features.append(feature)

        feature = request.form.get('polyuria')
        features.append(feature)

        feature = request.form.get('family_history')
        features.append(feature)

        feature = request.form.get('mucoid_sputum')
        features.append(feature)

        feature = request.form.get('rusty_sputum')
        features.append(feature)

        feature = request.form.get('lack_of_concentration')
        features.append(feature)

        feature = request.form.get('visual_disturbances')
        features.append(feature)

        feature = request.form.get('receiving_blood_transfusion')
        features.append(feature)

        feature = request.form.get('coma')
        features.append(feature)

        feature = request.form.get('stomach_bleeding')
        features.append(feature)

        feature = request.form.get('distention_of_abdomen')
        features.append(feature)

        feature = request.form.get('history_of_alcohol_consumption')
        features.append(feature)

        feature = request.form.get('fluid_overload')
        features.append(feature)

        feature = request.form.get('blood_in_sputum')
        features.append(feature)

        feature = request.form.get('prominent_veins_on_calf')
        features.append(feature)

        feature = request.form.get('palpitations')
        features.append(feature)

        feature = request.form.get('yellow_crust_ooze')
        features.append(feature)

        feature = request.form.get('painful_walking')
        features.append(feature)

        feature = request.form.get('pus_filled_pimples')
        features.append(feature)

        feature = request.form.get('blackheads')
        features.append(feature)

        feature = request.form.get('scurring')
        features.append(feature)

        feature = request.form.get('skin_peeling')
        features.append(feature)

        feature = request.form.get('silver_like_dusting')
        features.append(feature)

        feature = request.form.get('small_dents_in_nails')
        features.append(feature)

        feature = request.form.get('inflammatory_nails')
        features.append(feature)

        feature = request.form.get('blister')
        features.append(feature)

        feature = request.form.get('red_sore_around_nose')
        features.append(feature)

        feature = request.form.get('yellow_crust_ooze')
        features.append(feature)


        # for i in range(1,133):
        #     feature = request.form.get('weight_loss')
        #     features.append(feature)
        # feature1 = request.form.get('weight_loss')
        # feature2 = request.form.get('weight_loss')
        # feature3 = request.form.get('weight_loss')
        # feature4 = request.form.get('weight_loss')

        # Make predictions based on the selected features
        # features = [feature1, feature2, feature3, feature4]

        # Make predictions based on the selected features
        prediction = model.predict(pd.DataFrame(features).T)
        print(features)
        return prediction[0]
    # Create a Pandas DataFrame from the symptoms
    # input_data = pd.DataFrame(symptoms, index=[0])
    # Convert the symptoms to numeric values (0 or 1)
    # input_data = input_data.apply(pd.to_numeric)
    # Make the prediction using the loaded model
    # prediction = model.predict(input_data)
    # Return the predicted disease prognosis
    #return jsonify({"prognosis": prediction[0]})
    # return prediction[0]


if __name__ == "__main__":
    app.run(debug=True)
