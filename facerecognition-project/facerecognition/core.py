from django.http import request
from django.shortcuts import render, redirect
from django.http import JsonResponse
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from django.middleware.csrf import get_token
from django.views.decorators.csrf import csrf_exempt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from skimage import io as skimage_io
from skimage.color import rgb2gray
from skimage.feature import hog
from imutils import face_utils
from itertools import islice
from multiprocessing import Pool, cpu_count
from pyswarm import pso
import datetime
import json
import base64
import os
import io
import cv2
import numpy as np
import pickle
import dlib
import face_recognition
import datetime
import requests as req


def train_face(request):
    starttime = datetime.datetime.now()
    train("../image_face/", n_neighbors=5)

    endtime = datetime.datetime.now()
    training_time = endtime - starttime

    return render(request, 'success.html')


def process_images(image_paths):
    results = []
    for img_path, class_dir in image_paths:
        image = face_recognition.load_image_file(img_path)
        face_bounding_boxes = face_recognition.face_locations(
            image, number_of_times_to_upsample=1, model="cnn"
        )

        if len(face_bounding_boxes) == 1:
            face_encodings = face_recognition.face_encodings(
                image, known_face_locations=face_bounding_boxes, num_jitters=1
            )
            if face_encodings:
                results.append((face_encodings[0], class_dir))
    return results


def chunks(iterable, size):
    """Yield successive n-sized chunks from iterable."""
    iterator = iter(iterable)
    for first in iterator:  # stops when 'iterator' is depleted
        yield [first] + list(islice(iterator, size - 1))


def train(
    train_dir,
    model_save_path="trained_knn_model.clf",
    n_neighbors=5,
    metric="euclidean",
):
    X = []
    y = []

    image_paths = []
    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue

        for img_file in os.listdir(os.path.join(train_dir, class_dir)):
            img_path = os.path.join(train_dir, class_dir, img_file)
            image_paths.append((img_path, class_dir))

    # Use multiprocessing to process images in parallel
    with Pool(cpu_count()) as pool:
        for chunk in chunks(image_paths, 5):
            results = pool.apply(process_images, args=(chunk,))
            for face_encoding, class_dir in results:
                if face_encoding is not None:
                    X.append(face_encoding)
                    y.append(class_dir)

                    print(X)
                    print(y)

    best_hyperparameters = optimize_knn_hyperparameters(X, y)
    n_neighbors, metric = best_hyperparameters

    print(
        "Optimal hyperparameters: n_neighbors={}, metric={}".format(n_neighbors, metric)
    )

    metric_algo = "euclidean"
    psotrain("../image_face/", n_neighbors=int(round(n_neighbors)), metric=metric_algo)

    return True


def psotrain(
    train_dir,
    model_save_path="trained_knn_model.clf",
    n_neighbors=5,
    metric="euclidean",
):
    X = []
    y = []

    image_paths = []
    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue

        for img_file in os.listdir(os.path.join(train_dir, class_dir)):
            img_path = os.path.join(train_dir, class_dir, img_file)
            image_paths.append((img_path, class_dir))

    # Use multiprocessing to process images in parallel
    with Pool(cpu_count()) as pool:
        for chunk in chunks(image_paths, 5):
            results = pool.apply(process_images, args=(chunk,))
            for face_encoding, class_dir in results:
                if face_encoding is not None:
                    X.append(face_encoding)
                    y.append(class_dir)

                    print(X)
                    print(y)

    knn_clf = KNeighborsClassifier(
        n_neighbors=n_neighbors, weights="distance", metric=metric
    )
    knn_clf.fit(X, y)

    with open(model_save_path, "wb") as f:
        pickle.dump(knn_clf, f)

    return knn_clf


def optimize_knn_hyperparameters(X, y):
    def objective_function(hyperparameters):
        n_neighbors, metric = hyperparameters
        knn = KNeighborsClassifier(n_neighbors=int(n_neighbors), metric="euclidean")
        knn.fit(X, y)
        return -knn.score(X, y)  # We aim to maximize the negative accuracy

    valid_metrics = [
        "canberra",
        "p",
        "dice",
        "cityblock",
        "minkowski",
        "haversine",
        "nan_euclidean",
        "l2",
        "yule",
        "hamming",
        "euclidean",
        "sqeuclidean",
        "chebyshev",
        "pyfunc",
        "infinity",
        "seuclidean",
        "sokalsneath",
        "mahalanobis",
        "cosine",
        "russellrao",
        "jaccard",
        "l1",
        "rogerstanimoto",
        "correlation",
        "manhattan",
        "sokalmichener",
        "braycurtis",
        "precomputed",
    ]

    lb = [1, 0]  # Lower bounds for hyperparameters, where 0 corresponds to 'canberra'
    ub = [20, len(valid_metrics) - 1]  # Upper bounds for hyperparameters

    best_hyperparameters, _ = pso(objective_function, lb, ub, swarmsize=10, maxiter=100)
    return best_hyperparameters
