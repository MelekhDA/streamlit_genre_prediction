import pickle
from time import time

import streamlit as st

PATH_VECTORIZER = 'models/vectorizer.pkl'
PATH_MLB = 'models/multi_label_binarizer.pkl'
PATH_LOGREG = 'models/ovr_logreg.pkl'

st.header('Genre prediction')


def load_model(path, mode='rb'):
    with open(path, mode=mode) as fio:
        return pickle.load(fio)


def min_max_scaler(genres_proba: list) -> list:
    min_value = min(genres_proba, key=lambda x: x[1])[1]
    max_value = max(genres_proba, key=lambda x: x[1])[1]
    distance = max_value - min_value
    return [[genre, (proba - min_value) / distance] for genre, proba in genres_proba]


class Classifier:

    def __init__(self):
        self.vectorizer = load_model(PATH_VECTORIZER)
        self.mlb = load_model(PATH_MLB)
        self.logreg = load_model(PATH_LOGREG)

    def predict(self, text: str) -> list:
        vectorized_text = self.vectorizer.transform([text])
        predict = self.logreg.predict(vectorized_text)[0]
        return self.mlb.classes_[predict == 1]

    def predict_proba(self, text: str) -> list:
        vectorized_text = self.vectorizer.transform([text])
        predict = self.logreg.predict_proba(vectorized_text)[0]
        return sorted(zip(self.mlb.classes_, predict), key=lambda x: x[1], reverse=True)


time_from = time()
classifier = Classifier()
st.write(f'classifier loaded in {round(time() - time_from, 4)} seconds')

text = st.text_area("Enter text")

genres = classifier.predict(text)
genres_proba = classifier.predict_proba(text)

if len(genres) != 0:
    st.write(', '.join(genres))
elif len(text) > 0:
    genres_proba = min_max_scaler(genres_proba)
    genres_proba = [genre for genre, proba in genres_proba if proba >= 0.9]
    st.write('**No matching genre found**')
    st.write(f"Possibly genre(s): {', '.join(genres_proba)}")
