from django.shortcuts import render, redirect
from django.urls import reverse_lazy
from .models import Articles
from .forms import ArticlesForm
from django.views.generic import DetailView, UpdateView, DeleteView

def news_home(request):
    news = Articles.objects.order_by('-date')
    return render(request, 'news/news_home.html', {'news': news})

class NewsDetailView(DetailView):
    model = Articles
    template_name = 'news/details_view.html'
    context_object_name = 'article'

class NewsUpdateView(UpdateView):
    model = Articles
    template_name = 'news/create.html'
    form_class = ArticlesForm

class NewsDeleteView(DeleteView):
    model = Articles
    template_name = 'news/news_delete.html'
    success_url = reverse_lazy('news_home')

def create(request):
    form = ArticlesForm()
    error = ''
    if request.method == 'POST':
        form = ArticlesForm(request.POST)
        if form.is_valid():
            new_article = form.save()
            return redirect('news_detail', pk=new_article.pk)
        error = 'Форма была неверной'
    data = {
        'form': form,
        'error': error
    }
    return render(request, 'news/create.html', data)
