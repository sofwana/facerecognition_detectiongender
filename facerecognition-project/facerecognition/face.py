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
from pyswarm import pso
import json
import base64
import os
import io
import numpy as np
import pickle
import face_recognition
import requests as req

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}


# Start to predict the gender
def predict_gender(X_img_path, knn_clf, distance_threshold=0.7):
    if (
        not os.path.isfile(X_img_path)
        or os.path.splitext(X_img_path)[1][1:] not in ALLOWED_EXTENSIONS
    ):
        raise Exception("Invalid image path: {}".format(X_img_path))

    with open(knn_clf, "rb") as f:
        knn_clf = pickle.load(f)

    X_img = face_recognition.load_image_file(X_img_path)
    X_face_locations = face_recognition.face_locations(X_img)

    if len(X_face_locations) == 0:
        return []

    faces_encodings = face_recognition.face_encodings(
        X_img, known_face_locations=X_face_locations
    )
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=13)
    are_matches = [
        closest_distances[0][i][0] <= distance_threshold
        for i in range(len(X_face_locations))
    ]

    recognized_names = []
    for pred, rec in zip(knn_clf.predict(faces_encodings), are_matches):
        if rec:
            recognized_names.append(pred)
        else:
            recognized_names.append("unknown")

    return recognized_names


# Recognize
def recognize(request):
    data = request.body

    if request.method == "POST":
        # Get the image data from the POST request
        data = request.body

        data = json.loads(data.decode("utf-8"))
        frame = data.get("frame", None)
        try:
            # Specify the directory where you want to save the images
            save_directory = (
                "../image_face_test/"  # Use the absolute path to the directory
            )

            # Ensure the directory exists, creating it if necessary
            os.makedirs(save_directory, exist_ok=True)

            # Decode the Base64 data and save it as an image file
            content_type, image_data = frame.split(";base64,")
            image_format = content_type.split("/")[-1]
            filename = f"face_test.{image_format}"
            image_path = os.path.join(save_directory, filename)

            with open(image_path, "wb") as image_file:
                image_file.write(base64.b64decode(image_data))

            image_path = os.path.join(save_directory, "face_test.jpeg")

            res = predict_gender(image_path, knn_clf="trained_knn_model.clf")

            print(res)

            data = {"gender": res}

            return JsonResponse(data)

        except Exception as e:
            print(e)
            response_data = {"error": str(e)}
            return JsonResponse(response_data, status=400)

    return JsonResponse({"error": "Invalid request method"}, status=400)
