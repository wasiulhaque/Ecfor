from django.shortcuts import render, redirect
from django.http import HttpResponse

# Create your views here.


def registerPage(request):
    context = {}
    return render(request, 'register.html', context)


def loginPage(request):
    context = {}
    return render(request, 'login.html', context)
