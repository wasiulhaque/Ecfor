from django.urls import path
from Authentication import views

app_name = 'Authentication'

urlpatterns = [
    path('register/', views.registerPage, name="register"),
    path('login/', views.loginPage, name="login"),
    path('login/', views.logoutUser, name="logout"),

]
