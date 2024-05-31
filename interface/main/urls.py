from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('about/', views.about_view, name='about'),
    path('feedback/', views.feedback_view, name='feedback'),
    path('upload/', views.upload_and_display_files, name='upload_and_display'),]
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)