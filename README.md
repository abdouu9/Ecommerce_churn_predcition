# E-commerce Churn Prediction Streamlit App

This Streamlit application allows you to predict customer churn probability, identify risk segments, analyze top drivers, and get actionable recommendations.

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Save the model from your notebook:**
   - After training `model_xgb` in your notebook, run this code in a new cell:
   ```python
   import joblib
   
   # Save model with protocol 4 to avoid numpy random state issues
   joblib.dump(model_xgb, 'churn_model.pkl', protocol=4)
   print("✅ Model saved as 'churn_model.pkl'")
   
   # Save feature info
   feature_info = {
       'cat_cols': cat_cols,
       'num_cols': num_cols,
       'all_features': list(X.columns)
   }
   joblib.dump(feature_info, 'feature_info.pkl')
   print("✅ Feature info saved as 'feature_info.pkl'")
   ```
   - This will create:
     - `churn_model.pkl` - The trained XGBoost model with preprocessing pipeline
     - `feature_info.pkl` - Feature information for reference

3. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

## Features

- **Customer Input Form**: Enter customer features in the sidebar
- **Churn Probability**: Get the probability of churn (0-100%)
- **Risk Segmentation**: Automatically categorizes customers as Low, Medium, or High risk
- **Top Drivers**: Shows the top 5 factors influencing the churn prediction with SHAP values
- **SHAP Visualizations**: Interactive waterfall and bar plots showing feature contributions
- **Actionable Recommendations**: Provides specific solutions based on the top drivers

## Usage

1. Fill in all customer information in the sidebar
2. Click "Predict Churn"
3. View the results:
   - Churn probability and risk segment
   - Top drivers analysis
   - SHAP visualizations
   - Recommended actions

## Model Details

- **Algorithm**: XGBoost Classifier
- **Preprocessing**: 
  - Iterative Imputer for missing values
  - Standard Scaler for numerical features
  - One-Hot Encoder for categorical features
- **Features**: 18 customer features including tenure, satisfaction score, order history, and demographics

