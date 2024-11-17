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

i

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


