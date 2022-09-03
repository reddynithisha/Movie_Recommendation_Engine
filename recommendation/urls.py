from django.urls import path
app_name = 'recommendation'

from . import views

urlpatterns = [

   
    path('',views.home,name='home'), 
    path('details',views.movie_details,name='movie_details'),
    path('autosuggestion',views.autosuggestion,name='autosuggestion'),
    

]