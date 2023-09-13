from django.shortcuts import render
from django.http import HttpResponse
import os
import json

from .Index import Index

files_index = Index()
files_index.load_data()


def index(request):
    return render(request, "search/index.html")


def search(request):
    if request.method == 'POST':
        # Get search query
        q = request.POST.get('q')

        results = files_index.search(query=q)

        return render(request, "search/results.html", {
            "search_query": q,
            "results": results,
        })