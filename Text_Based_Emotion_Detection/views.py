from django.shortcuts import render
from users.forms import UserRegistrationForm
from .bert_model import predict_emotion


def index(request):
    return render(request, 'index.html', {})

def AdminLogin(request):
    return render(request, 'AdminLogin.html', {})

def UserLogin(request):
    return render(request, 'UserLogin.html', {})


def UserRegister(request):
    form = UserRegistrationForm()
    return render(request, 'UserRegistrations.html', {'form': form})

# ✅ New view for emotion detection
def EmotionDetection(request):
    emotion = None
    user_text = ""
    
    if request.method == "POST":
        user_text = request.POST.get("user_text")
        print("Received text:", user_text)  # <-- check terminal
        if user_text:
            emotion = predict_emotion(user_text)
            print("Predicted emotion:", emotion)  # <-- check terminal

    return render(request, 'EmotionDetection.html', {'emotion': emotion, 'user_text': user_text})
