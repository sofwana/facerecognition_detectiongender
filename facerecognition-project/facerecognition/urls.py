from django.contrib import admin
from django.urls import path

from . import views
from . import core
from . import face

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.index),
    path('detection/', views.detection),
    path('recognize/', face.recognize, name='recognize'),
    path('train/', core.train_face, name='train_face'),
    path('training-data/', views.view_train, name='train_page'),
    path('training-data/success/', views.success, name='success'),
    path('sofwan', views.success, name='Sofwan')
]
