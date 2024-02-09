import requests
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.optimizers import Adam
import numpy as np
import io
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your custom model and compile it
custom_model = load_model('custom_flower_recognition_model.h5')
custom_model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Define the path to your dataset
dataset_path = 'flowers'
train_data_dir = os.path.join(dataset_path, 'train')

# Extract class labels from the directory structure
class_labels = sorted(os.listdir(train_data_dir))

# Google Custom Search API Key
google_api_key = 'AIzaSyDEfbCqgoiz2CS9CcFvHgHioCdckm9No3M'


def process_predictions(predictions, class_labels):
    if len(predictions.shape) == 2 and predictions.shape[0] == 1:
        predictions = predictions[0]

    predicted_class_index = np.argmax(predictions)
    predicted_class_label = class_labels[predicted_class_index]
    confidence = predictions[predicted_class_index]

    return predicted_class_label, confidence


def get_flower_details_from_google(query):
    # Construct a search query based on the recognized flower name
    search_query = f'{query} flower details'

    # Make a request to the Google Custom Search JSON API
    params = {
        'q': search_query,
        'key': google_api_key,
        'cx': '972aec82cb22c4d7a',  # Replace with your custom search engine ID
    }

    response = requests.get('https://www.googleapis.com/customsearch/v1', params=params)
    result = response.json()

    # Extract relevant details from the API response
    # Adjust this part based on the structure of the API response
    details = result.get('items', [])

    return details


def process_image(file):
    try:
        img_bytes = io.BytesIO(file.file.read())

        img = image.load_img(img_bytes, target_size=(299, 299))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        predictions = custom_model.predict(img_array)
        predicted_class, confidence = process_predictions(predictions, class_labels)

        flower_details_from_google = get_flower_details_from_google(predicted_class)

        result = {
            'flower_name': predicted_class,
            'description': flower_details_from_google,
            'confidence': float(confidence),
        }

        return result

    except Exception as e:
        print(f'Error processing image {e}')
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/uploadFile/")
async def create_upload_file(file: UploadFile = File(...)):
    result = process_image(file)
    return JSONResponse(content=result)
