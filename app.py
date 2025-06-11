from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
import os
from datetime import datetime

app = Flask(__name__)

# Global variables for model components
model = None
scaler = None
feature_names = None
model_info = None


def load_model_components():
    """Load model, scaler và feature names"""
    global model, scaler, feature_names, model_info

    try:
        # Load model info
        with open('model_info.pkl', 'rb') as file:
            model_info = pickle.load(file)

        # Load model
        model_filename = f'vietnam_housing_price_model_{model_info["model_name"].lower().replace(" ", "_")}.pkl'
        with open(model_filename, 'rb') as file:
            model = pickle.load(file)

        # Load scaler nếu có
        if model_info['uses_scaler']:
            with open('scaler.pkl', 'rb') as file:
                scaler = pickle.load(file)

        # Load feature names
        with open('feature_names.pkl', 'rb') as file:
            feature_names = pickle.load(file)

        print(f"✅ Model loaded successfully: {model_info['model_name']}")
        print(f"✅ Number of features: {len(feature_names)}")
        print(f"✅ Uses scaler: {model_info['uses_scaler']}")

        return True

    except Exception as e:
        print(f"❌ Error loading model components: {e}")
        return False


def prepare_input_data(form_data):
    """Chuẩn bị dữ liệu đầu vào cho model"""
    try:
        # Tạo DataFrame với tất cả features, fill với 0
        input_data = pd.DataFrame(0, index=[0], columns=feature_names)

        # Điền dữ liệu từ form
        for key, value in form_data.items():
            if key in feature_names:
                input_data[key] = value
            else:
                # Xử lý categorical variables (one-hot encoded)
                # Tìm column tương ứng trong feature_names
                matching_cols = [col for col in feature_names if col.startswith(f"{key}_")]
                if matching_cols and value:
                    # Reset tất cả columns của category này về 0
                    for col in matching_cols:
                        input_data[col] = 0
                    # Set column được chọn thành 1
                    target_col = f"{key}_{value}"
                    if target_col in feature_names:
                        input_data[target_col] = 1

        # Tính toán derived features nếu có
        if 'Area' in form_data and 'Bedrooms' in form_data:
            area = float(form_data['Area'])
            bedrooms = float(form_data['Bedrooms'])
            if 'Area_per_Bedroom' in feature_names:
                input_data['Area_per_Bedroom'] = area / (bedrooms + 1)

        if 'Year_Built' in form_data:
            year_built = float(form_data['Year_Built'])
            if 'House_Age' in feature_names:
                input_data['House_Age'] = 2024 - year_built

        if 'Bedrooms' in form_data and 'Bathrooms' in form_data:
            bedrooms = float(form_data['Bedrooms'])
            bathrooms = float(form_data['Bathrooms'])
            if 'Total_Rooms' in feature_names:
                input_data['Total_Rooms'] = bedrooms + bathrooms

        return input_data

    except Exception as e:
        raise Exception(f"Error preparing input data: {str(e)}")


@app.route('/')
def home():
    """Trang chủ"""
    return render_template('index.html', model_info=model_info)


@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint để dự đoán giá nhà"""
    try:
        if model is None:
            return jsonify({
                'success': False,
                'error': 'Model not loaded. Please check server configuration.'
            })

        # Lấy dữ liệu từ request
        form_data = request.get_json()

        if not form_data:
            return jsonify({
                'success': False,
                'error': 'No input data provided.'
            })

        # Chuẩn bị dữ liệu
        input_df = prepare_input_data(form_data)

        # Dự đoán
        if scaler is not None:
            # Scale dữ liệu nếu cần
            input_scaled = scaler.transform(input_df)
            prediction = model.predict(input_scaled)[0]
        else:
            # Dự đoán trực tiếp (Random Forest)
            prediction = model.predict(input_df)[0]

        # Đảm bảo giá trị dương
        prediction = max(0, prediction)

        formatted_price = f"{prediction:,.2f} tỷ VND"

        return jsonify({
            'success': True,
            'predicted_price': float(prediction),
            'formatted_price': formatted_price,
            'model_used': model_info['model_name']
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


@app.route('/model_info')
def get_model_info():
    """API để lấy thông tin model"""
    if model_info:
        return jsonify(model_info)
    else:
        return jsonify({'error': 'Model info not available'})


@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat()
    })


if __name__ == '__main__':
    print("🚀 Starting Vietnam Housing Price Predictor...")

    # Load model components
    if load_model_components():
        print("🌟 Server ready!")
        print("📍 Access the application at: http://localhost:5000")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Failed to load model. Please check your model files.")
        print("Required files:")
        print("  - model_info.pkl")
        print("  - vietnam_housing_price_model_*.pkl")
        print("  - feature_names.pkl")
        print("  - scaler.pkl (if model uses scaling)")