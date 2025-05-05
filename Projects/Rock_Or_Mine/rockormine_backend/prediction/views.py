from django.shortcuts import render
from django.http import JsonResponse, HttpResponseBadRequest
import os
import joblib
import numpy as np

# Load the model once when the module is imported
model_path = os.path.join(os.path.dirname(__file__), 'logistic_model.joblib')
model = joblib.load(model_path)

def predict(request):
    if request.method == 'POST' and request.headers.get('x-requested-with') == 'XMLHttpRequest':
        try:
            # Expecting JSON data with 'input' key containing list of features
            import json
            data = json.loads(request.body).get('input')
            if not data:
                return JsonResponse({'error': "Missing 'input' data"}, status=400)
            input_array = np.array(data).reshape(1, -1)
            prediction = model.predict(input_array)
            result = 'Rock' if prediction[0] == 'R' else 'Mine'
            return JsonResponse({'prediction': result})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)
    elif request.method == 'POST':
        # Handle form submission from front-end
        input_str = request.POST.get('input')
        if not input_str:
            return render(request, 'prediction/index.html', {'error': "Please enter input data."})
        try:
            input_list = [float(x.strip()) for x in input_str.split(',')]
            if len(input_list) != 60:
                return render(request, 'prediction/index.html', {'error': "Please enter exactly 60 feature values."})
            input_array = np.array(input_list).reshape(1, -1)
            prediction = model.predict(input_array)
            result = 'Rock' if prediction[0] == 'R' else 'Mine'
            return render(request, 'prediction/index.html', {'prediction': result})
        except Exception as e:
            return render(request, 'prediction/index.html', {'error': f"Error processing input: {str(e)}"})
    else:
        return render(request, 'prediction/index.html')
