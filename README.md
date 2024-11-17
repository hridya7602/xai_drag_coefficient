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

