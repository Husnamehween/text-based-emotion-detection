from ast import alias
from concurrent.futures import process
from django.shortcuts import render

# Create your views here.
from django.shortcuts import render, HttpResponse
from django.contrib import messages

import Text_Based_Emotion_Detection

from .forms import UserRegistrationForm
from .models import UserRegistrationModel
from django.conf import settings
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import datetime as dt
from sklearn import preprocessing, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import classification_report

#new

import os
import re
import pickle
import tensorflow as tf

from tensorflow.keras.preprocessing.sequence import pad_sequences
#newend
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Create your views here.

def UserRegisterActions(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            print('Data is Valid')
            form.save()
            messages.success(request, 'You have been successfully registered')
            form = UserRegistrationForm()
            return render(request, 'UserRegistrations.html', {'form': form})
        else:
            messages.success(request, 'Email or Mobile Already Existed')
            print("Invalid form")
    else:
        form = UserRegistrationForm()
    return render(request, 'UserRegistrations.html', {'form': form})

def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')
        print("Login ID = ", loginid, ' Password = ', pswd)
        try:
            check = UserRegistrationModel.objects.get(
                loginid=loginid, password=pswd)
            status = check.status
            print('Status is = ', status)
            if status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                print("User id At", check.id, status)
                return render(request, 'users/UserHomePage.html', {})
            else:
                messages.success(request, 'Your Account Not at activated')
                return render(request, 'UserLogin.html')
        except Exception as e:
            print('Exception is ', str(e))
            pass
        messages.success(request, 'Invalid Login id and password')
    return render(request, 'UserLogin.html', {})



def UserHome(request):
    return render(request, 'users/UserHomePage.html', {})

def DatasetView(request):
    path = settings.MEDIA_ROOT + "//" + 'balanced_data.csv'
    df = pd.read_csv(path)
    df = df.to_html
    return render(request, 'users/viewdataset.html', {'data': df})

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
path = settings.MEDIA_ROOT + "//" + 'balanced_data.csv'
df = pd.read_csv(path)
value_counts = df['label'].value_counts()
# print(value_counts)
# import matplotlib.pyplot as plt
# # Plotting the bar plot
# value_counts.plot(kind='bar')
# # Adding labels and title
# plt.xlabel('Categories')
# plt.ylabel('Counts')
# plt.title('Value Counts of Your Column')
# # Displaying the plot
# plt.show()
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)
# Vectorize the tweets using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=10000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)
# Train a Naive Bayes classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_tfidf, y_train)

def training(request):  
    # Adding labels and title
    plt.xlabel('Categories')
    plt.ylabel('Counts')
    plt.title('Value Counts of Your Column')
    # Displaying the plot
    plt.show()    
    predictions = nb_classifier.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy:.2f}")  

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, predictions)
    print("Naïve Bayes Confusion Matrix:")
    print(cm)
    
        
    nb = classification_report(y_test,predictions,output_dict=True)
    nb = pd.DataFrame(nb).transpose()    
    nb = pd.DataFrame(nb)
    return render(request,"users/training.html",{'nb':nb.to_html,'acc':accuracy})
    
'''
a classification label, with possible values including 
sadness (0), joy (1), love (2), anger (3), fear (4). happy(5).
'''
#new 
nb_classifier.fit(X_train_tfidf, y_train)


# Load the existing LSTM model and recovered tokenizer
BASE_DIR = settings.BASE_DIR

LSTM_MODEL_PATH = os.path.join(BASE_DIR, "lstm_emotion_model.h5")
LSTM_TOKENIZER_PATH = os.path.join(BASE_DIR, "lstm_tokenizer.pkl")

lstm_model = tf.keras.models.load_model(LSTM_MODEL_PATH)

with open(LSTM_TOKENIZER_PATH, "rb") as file:
    lstm_tokenizer = pickle.load(file)

MAX_LEN = 80

EMOTION_LABELS = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise",
}

#new

# Load the fine-tuned RoBERTa model
ROBERTA_MODEL_PATH = os.path.join(
    settings.MEDIA_ROOT, "fine_tuned_roberta"
)

roberta_tokenizer = AutoTokenizer.from_pretrained(
    ROBERTA_MODEL_PATH
)

roberta_model = AutoModelForSequenceClassification.from_pretrained(
    ROBERTA_MODEL_PATH
)
roberta_model.eval()


def predict_with_roberta(text):
    inputs = roberta_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = roberta_model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1)
        predicted_index = int(probabilities.argmax(dim=1).item())

    return {
        "emotion": EMOTION_LABELS[predicted_index],
        "confidence": round(
            float(probabilities[0, predicted_index].item()) * 100, 2
        )
    }
#newend

def clean_lstm_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def predict_with_lstm(text):
    cleaned_text = clean_lstm_text(text)
    sequence = lstm_tokenizer.texts_to_sequences([cleaned_text])

    padded_sequence = pad_sequences(
        sequence,
        maxlen=MAX_LEN,
        padding="post",
        truncating="post"
    )

    probabilities = lstm_model.predict(padded_sequence, verbose=0)
    predicted_index = int(probabilities[0].argmax())

    return {
        "emotion": EMOTION_LABELS[predicted_index],
        "confidence": round(float(probabilities[0][predicted_index]) * 100, 2),
    }
#newend

def prediction(request):
    if request.method == 'POST':
        single_tweet = request.POST.get('tweets') 
        print(single_tweet)      
        single_tweet_tfidf = tfidf_vectorizer.transform([single_tweet])
        print('manohar',single_tweet_tfidf)
        # Make prediction
        single_prediction = nb_classifier.predict(single_tweet_tfidf)
        print(single_prediction)
        # Print prediction
        print(f'Tweet: {single_tweet} - Predicted Emotion: {single_prediction[0]}')
        if single_prediction[0] == 0:
            single_prediction='sadness'
        elif single_prediction[0] == 1:
            single_prediction='joy'
        elif single_prediction[0] == 2:
            single_prediction='love'
        elif single_prediction[0] == 3:
            single_prediction='anger'
        elif single_prediction[0] == 4:
            single_prediction='fear'
        elif single_prediction[0] == 5:
            single_prediction='surprise'
        return render(request, 'users/predictForm.html', {'output':single_prediction})
    return render(request, 'users/predictForm.html', {})
        
 #new
   
def compare_emotions(request):
    context = {}

    if request.method == "POST":
        text = request.POST.get("tweets", "").strip()

        if text:
            # Naive Bayes prediction
            nb_features = tfidf_vectorizer.transform([text])
            nb_index = int(nb_classifier.predict(nb_features)[0])

            context["text"] = text
            context["nb_emotion"] = EMOTION_LABELS.get(
                nb_index, f"Unknown label: {nb_index}"
            )

            # LSTM prediction
            context["lstm_result"] = predict_with_lstm(text)

    return render(request, "users/compare_emotions.html", context)



def compare_emotions(request):
    context = {}

    if request.method == "POST":
        text = request.POST.get("tweets", "").strip()

        if text:
            # Naive Bayes
            nb_features = tfidf_vectorizer.transform([text])
            nb_index = int(nb_classifier.predict(nb_features)[0])

            context["text"] = text
            context["nb_emotion"] = EMOTION_LABELS.get(
                nb_index, f"Unknown label: {nb_index}"
            )

            # LSTM
            context["lstm_result"] = predict_with_lstm(text)

            # Fine-tuned RoBERTa
            context["roberta_result"] = predict_with_roberta(text)

    return render(request, "users/compare_emotions.html", context)

#newend     
            