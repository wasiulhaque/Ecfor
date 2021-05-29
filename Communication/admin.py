from django.contrib import admin
from django.contrib.admin.decorators import register
from Communication.models import testDb
# Register your models here.

admin.site.register(testDb)
