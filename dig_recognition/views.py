from django.shortcuts import render
from django.http import JsonResponse
import numpy as np
import tensorflow as tf
import json
from django.http import JsonResponse
import cv2
import numpy as np
import base64


# Load the pre-trained model
model = tf.keras.models.load_model("digit_detector.keras")

def process_image(image_data):

    # Remove the "data:image/png;base64," part from the beginning of the string
    image_data = image_data.split(",")[1]

    # Decode the Base64 string
    decoded_image_data = base64.b64decode(image_data)

    # Convert the decoded bytes into a numpy array
    np_arr = np.frombuffer(decoded_image_data, dtype="uint8")

    # Read the image
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #resize image to 28x28 pixels
    img = cv2.resize(img, (28,28), interpolation=cv2.INTER_AREA)
    #invert from black writing on white to white writing on black (MNIST is in that format)
    img = np.invert(img)
    #normalize pixel values
    img = tf.keras.utils.normalize(img, axis=1)
    #prepare for convolutional operations by making correct dimensions
    img = np.array(img).reshape(-1, 28, 28, 1)

    return img


#gets the image data url and sends back predicted digit
def predict_digit(request):
    if request.method == "GET":
        return render(request, "predict.html")
    
    if request.method == 'POST':
        try:
            # Parse the JSON data from the request
            data = json.loads(request.body)
            image_data = data.get("image")

            #prepare the image for the ml model
            img = process_image(image_data)

            #get the response of the model
            prediction = model.predict(img)

            predicted_digit = np.argmax(prediction, axis=1)[0]

            # Return the prediction as a JSON response
            return JsonResponse({'prediction': int(predicted_digit)})
        
        except Exception as e:
            # Handle any errors that occur
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Invalid request method.'}, status=400)
