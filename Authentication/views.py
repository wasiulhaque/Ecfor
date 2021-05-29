from django import forms
from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.forms import inlineformset_factory
from django.views.decorators.csrf import csrf_exempt, csrf_protect
from django.contrib.auth.forms import UserCreationForm

from django.contrib import messages

from django.contrib.auth import authenticate, login, logout

from Authentication.forms import CreateUserForm
# Create your views here.

# p1q2r3s4


@csrf_exempt
def registerPage(request):
    form = CreateUserForm()

    if request.method == 'POST':
        form = CreateUserForm(request.POST)  # render the form data
        if form.is_valid():
            form.save()
            user = form.cleaned_data.get('username')
            messages.success(
                request, 'Account is created successfully for ' + user)
            return redirect('Authentication:login')

    context = {'form': form}
    return render(request, 'register.html', context=context)


def loginPage(request):
    context = {}
    return render(request, 'login.html', context)
