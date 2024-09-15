from django.http import request
from django.shortcuts import render, redirect
from django.contrib import messages
import requests as req
import json
from django.middleware.csrf import get_token

def index(request):
    return render(request, 'index.html')

def view_train(request):
    return render(request, 'train.html')

def success(request):
    return render(request, 'success.html')
    
def success(request):
    return render(request, 'sofwan.html')

def detection(request):
    csrfToken = get_token(request)

    context = {
        'csrf' : csrfToken
    }

    return render(request, 'detection.html', context)