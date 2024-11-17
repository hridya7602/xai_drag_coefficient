# xai_drag_coefficient

## Overview
`xai_drag_coefficient` is a machine learning project that integrates Explainable Artificial Intelligence (XAI) to predict car drag coefficients. The project uses deep learning techniques to predict drag coefficients from 2D renderings of 3D car meshes and utilizes XAI methods like SHAP for interpretability, helping engineers understand the key features influencing aerodynamic performance.

## Features
- **3D to 2D Conversion**: Converts 3D car meshes into 2D orthographic views with surface normals and depth information.
- **Fast Predictions**: A surrogate deep learning model (e.g., attn-ResNeXt) predicts the drag coefficient efficiently.
- **XAI Interpretability**: SHAP (or LIME) provides explanations on the features affecting drag coefficient predictions.
- **User Interface**: A web interface to upload 3D models, view predictions, and interpret results.

## Requirements

### Python Dependencies
- Python 3.7 or higher
- PyTorch or TensorFlow for deep learning
- SHAP or LIME for interpretability
- Open3D or PCL for 3D mesh processing
- Flask/Django for API backend
- React/Vue.js for frontend development
- MongoDB/PostgreSQL for database storage

### Install Dependencies
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/xai_drag_coefficient.git
   cd xai_drag_coefficient
Install Python dependencies:

bash
Copy code
pip install -r requirements.txt
For frontend dependencies (if using React or Vue.js):

bash
Copy code
cd frontend
npm install
Project Structure
bash
Copy code
xai_drag_coefficient/
│
├── backend/             # Backend code (Flask or Django API)
│   ├── app.py           # Main application
│   ├── model.py         # Model training and inference
│   ├── utils.py         # Helper functions for mesh processing
│   └── requirements.txt # Backend dependencies
│
├── frontend/            # Frontend code (React or Vue.js)
│   ├── src/
│   └── package.json     # Frontend dependencies
│
├── data/                # Datasets (3D car meshes)
│   └── mesh_samples/    # Example 3D car meshes
│
└── README.md            # This file
Usage
Step 1: Upload 3D Car Mesh
Users can upload a 3D car mesh (e.g., .obj or .stl file) through the web interface.

Step 2: Preprocess 3D Mesh into 2D Views
Once uploaded, the system processes the 3D model into six orthographic views, converting depth and surface normals into 2D representations. This is done using Open3D:

python
Copy code
import open3d as o3d

def process_3d_mesh(mesh_path):
    # Load the 3D mesh
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    
    # Convert the mesh to orthographic views
    orthographic_views = convert_to_orthographic_views(mesh)
    
    return orthographic_views
Step 3: Predict Drag Coefficient
The processed 2D renderings are fed into the surrogate deep learning model (e.g., attn-ResNeXt):

python
Copy code
import torch
from model import SurrogateModel

# Load the trained model
model = SurrogateModel()
model.load_state_dict(torch.load('drag_model.pth'))

# Perform inference
def predict_drag_coefficient(orthographic_views):
    with torch.no_grad():
        input_tensor = torch.tensor(orthographic_views)
        prediction = model(input_tensor)
        return prediction.item()
Step 4: Interpret Predictions with XAI (SHAP)
SHAP (SHapley Additive exPlanations) is used to explain the influence of each feature (e.g., shape, surface area) on the predicted drag coefficient:

python
Copy code
import shap

def explain_prediction(model, data):
    explainer = shap.Explainer(model)
    shap_values = explainer.shap_values(data)
    shap.initjs()
    shap.force_plot(shap_values[0], data[0], feature_names=["Feature1", "Feature2"])
Step 5: View Results
The drag coefficient prediction and the SHAP visual explanation will be displayed in the frontend:

js
Copy code
// Example of React component to display results
import React, { useState } from 'react';

function DragCoefficientPrediction() {
  const [dragCoefficient, setDragCoefficient] = useState(null);
  const [shapExplanation, setShapExplanation] = useState(null);

  const handleFileUpload = (file) => {
    // Call API to process and predict drag coefficient
    fetch('/api/predict', {
      method: 'POST',
      body: file
    })
    .then(response => response.json())
    .then(data => {
      setDragCoefficient(data.prediction);
      setShapExplanation(data.shap_explanation);
    });
  };

  return (
    <div>
      <input type="file" onChange={(e) => handleFileUpload(e.target.files[0])} />
      <div>Drag Coefficient: {dragCoefficient}</div>
      <div>{shapExplanation && <img src={shapExplanation} alt="SHAP Explanation" />}</div>
    </div>
  );
}

export default DragCoefficientPrediction;
Deployment
Run the Backend Server (Flask Example)
Start the backend server to handle requests:

bash
Copy code
python backend/app.py
Deploy Frontend
To run the frontend:

bash
Copy code
cd frontend
npm start
The frontend will be accessible at http://localhost:3000, and the backend API will be available at http://localhost:5000.

License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments
The project was supported by research in the field of aerodynamic design optimization.
Special thanks to contributors who helped with the dataset of 3D car meshes.
