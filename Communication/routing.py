from django.urls import path

from .views import WSConsumer       # .dot means current folder

ws_urlpatterns = [
    path('speak/realtime/', WSConsumer.as_asgi())
]
