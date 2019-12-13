from flask import Flask, request
from flask_restful import Api, Resource
from flask_cors import CORS
from sklearn.externals import joblib
import numpy as np
import pickle as p


app = Flask(__name__)
app.config['SECRET_KEY'] = 'blocking 4'

CORS(app)

api = Api(app)

modelfile = 'models/recruit.pkl'
model = joblib.load(open(modelfile, 'rb'))

class Predict(Resource):
    def post(self):

        try:
            json_data = request.get_json()

            qualification = json_data['qualification']
            target_met = json_data['target_met'] 
            foreign_schooled = json_data['foreign_schooled']
            previous_award = json_data['previous_award']
            class_of_degree = json_data['class_of_degree']
            interview_test = json_data['interview_test']
            fluency_confidence = json_data['fluency_confidence']

            prediction_data = [qualification, target_met, foreign_schooled, previous_award, class_of_degree,
                                interview_test, fluency_confidence]

            prediction = np.array2string(model.predict([prediction_data]))

            prediction = prediction.strip('[]')


            return {
                    'status code':'200',
                    'message':prediction
                    }, 200
        except:
            return {
                    'status code':'500',
                    'message':'Invalid Values'
                    }, 500


class Homepage(Resource):
    def get(self):
        return 'Hello'




#connect class with endpoint
api.add_resource(Predict, '/api/predict')
api.add_resource(Homepage, '/')

if __name__ == '__main__':
    app.run(debug=True)