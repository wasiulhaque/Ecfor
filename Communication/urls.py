from django.urls import path
from Communication import views

app_name = 'Communication'

urlpatterns = [
    path('index/', views.index, name='index'),
    path('listen/', views.listen, name='listen'),
    # path('speak/', views.speak, name='speak'),
    path('speak/', views.ML, name='speak'),
]
