from django.shortcuts import render
from django.http import HttpResponse

# Create your views here.


def index(request):
    diction = {'sample_text': 'This is a text'}
    return render(request, 'index.html', context=diction)
