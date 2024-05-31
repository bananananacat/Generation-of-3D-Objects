import os
from django.conf import settings
from django.shortcuts import render, redirect
from .utils import convert_photos_to_3d_model
from django.http import HttpResponse
from django.views.generic.edit import FormView
from .forms import UploadFileForm

def handle_uploaded_files(files):
    save_path = os.path.join(settings.MEDIA_ROOT, 'uploads')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for file in files:
        print(file)
        with open(os.path.join(save_path, file.name), 'wb+') as destination:
            for chunk in file.chunks():
                destination.write(chunk)
    return save_path

def upload_and_display_files(request):

    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            handle_uploaded_files(request.FILES.getlist('files'))
            return redirect('upload_and_display')
    else:
        form = UploadFileForm()

    return render(request, 'main/upload.html', {'form': form})

def about_view(request):
    return render(request, 'main/about.html')

def feedback_view(request):
    return render(request, 'main/feedback.html')
