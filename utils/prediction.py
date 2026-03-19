import os
import pickle
from utils.feature_extraction import extract_features
from utils.prediction import final_prediction


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

gender_model = pickle.load(open(os.path.join(BASE_DIR, "models/gender_model_xgb.pkl"), "rb"))
gender_le = pickle.load(open(os.path.join(BASE_DIR, "models/label_encoder.pkl"), "rb"))

emotion_model = pickle.load(open(os.path.join(BASE_DIR, "models/emotion_model.pkl"), "rb"))
emotion_le = pickle.load(open(os.path.join(BASE_DIR, "models/emotion_label_encoder.pkl"), "rb"))
def final_prediction(file_path):

    features = extract_features(file_path)
    features = features.reshape(1, -1)

    gender_pred = gender_model.predict(features)
    gender = gender_le.inverse_transform(gender_pred)[0]

    if gender != "female":
        return "Please upload a female voice sample"

    emotion_pred = emotion_model.predict(features)
    emotion = emotion_le.inverse_transform(emotion_pred)[0]

    return f"Emotion: {emotion}"