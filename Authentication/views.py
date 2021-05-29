from django import forms
from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.forms import inlineformset_factory
from django.views.decorators.csrf import csrf_exempt, csrf_protect
from django.contrib.auth.forms import UserCreationForm


from Authentication.forms import CreateUserForm
# Create your views here.


@csrf_exempt
def registerPage(request):
    form = CreateUserForm()

    if request.method == 'POST':
        form = CreateUserForm(request.POST)  # render the form data
        if form.is_valid():
            form.save()

    context = {'form': form}
    return render(request, 'register.html', context=context)


def loginPage(request):
    context = {}
    return render(request, 'login.html', context)
