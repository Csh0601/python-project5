import os
import sys
import json
import numpy as np
from pathlib import Path
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from .models import SearchRecord

# Add assignments path to sys.path
ASSIGNMENTS_PATH = Path(__file__).resolve().parent.parent.parent / 'assignments'
sys.path.insert(0, str(ASSIGNMENTS_PATH))

from dinov2_numpy import Dinov2Numpy
from preprocess_image import center_crop, resize_short_side

# Global variables for model and features
_vit_model = None
_gallery_features = None
_gallery_paths = None

def get_vit_model():
    """Lazy load ViT model"""
    global _vit_model
    if _vit_model is None:
        weights_path = ASSIGNMENTS_PATH / 'vit-dinov2-base.npz'
        weights = np.load(str(weights_path))
        _vit_model = Dinov2Numpy(weights)
    return _vit_model

def get_gallery_data():
    """Lazy load gallery features and paths"""
    global _gallery_features, _gallery_paths
    if _gallery_features is None:
        features_dir = Path(__file__).resolve().parent.parent / 'features'
        features_path = features_dir / 'gallery_features.npz'
        if features_path.exists():
            data = np.load(str(features_path), allow_pickle=True)
            _gallery_features = data['features']
            _gallery_paths = data['paths']
    return _gallery_features, _gallery_paths

def cosine_similarity(query_feat, gallery_feats):
    """Compute cosine similarity between query and gallery"""
    query_norm = query_feat / (np.linalg.norm(query_feat) + 1e-8)
    gallery_norms = gallery_feats / (np.linalg.norm(gallery_feats, axis=1, keepdims=True) + 1e-8)
    similarities = np.dot(gallery_norms, query_norm)
    return similarities

def index(request):
    """Main page"""
    history = []
    if request.user.is_authenticated:
        history = SearchRecord.objects.filter(user=request.user)[:5]
    return render(request, 'index.html', {'history': history})


def register_view(request):
    """User registration"""
    if request.user.is_authenticated:
        return redirect('index')
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect('index')
    else:
        form = UserCreationForm()
    return render(request, 'register.html', {'form': form})


def login_view(request):
    """User login"""
    if request.user.is_authenticated:
        return redirect('index')
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            return redirect('index')
    else:
        form = AuthenticationForm()
    return render(request, 'login.html', {'form': form})


def logout_view(request):
    """User logout"""
    logout(request)
    return redirect('index')

@csrf_exempt
def search(request):
    """Handle image search request"""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST method required'}, status=400)

    if 'image' not in request.FILES:
        return JsonResponse({'error': 'No image uploaded'}, status=400)

    # Save uploaded image temporarily
    uploaded_file = request.FILES['image']
    temp_path = Path(settings.MEDIA_ROOT) / 'temp_query.jpg'
    temp_path.parent.mkdir(parents=True, exist_ok=True)

    with open(temp_path, 'wb') as f:
        for chunk in uploaded_file.chunks():
            f.write(chunk)

    try:
        # Extract features from query image
        vit = get_vit_model()
        pixel_values = center_crop(str(temp_path))
        query_feat = vit(pixel_values)[0]

        # Get gallery features
        gallery_feats, gallery_paths = get_gallery_data()

        if gallery_feats is None:
            return JsonResponse({'error': 'Gallery features not found. Run extract_features.py first.'}, status=500)

        # Compute similarities and get top-10
        similarities = cosine_similarity(query_feat, gallery_feats)
        top_indices = np.argsort(similarities)[::-1][:10]

        results = []
        for idx in top_indices:
            results.append({
                'path': str(gallery_paths[idx]),
                'similarity': float(similarities[idx])
            })

        # Save search record for authenticated users
        if request.user.is_authenticated:
            # Save query image permanently
            import uuid
            query_filename = f"queries/{uuid.uuid4().hex}.jpg"
            query_save_path = Path(settings.MEDIA_ROOT) / query_filename
            query_save_path.parent.mkdir(parents=True, exist_ok=True)

            # Copy temp file to permanent location
            import shutil
            shutil.copy(str(temp_path), str(query_save_path))

            # Create search record
            SearchRecord.objects.create(
                user=request.user,
                query_image=query_filename,
                results=results
            )

        return JsonResponse({'results': results})

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

    finally:
        if temp_path.exists():
            temp_path.unlink()
