import os
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
import base64
from datetime import datetime
import json
import random

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'face_recognition_secret_key'
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Mock database (in-memory for demo)
users = []
next_user_id = 1

# Mock face recognition (for demo purposes)
class MockFaceRecognizer:
    def __init__(self):
        self.trained = False

    def extract_features(self, img_data):
        # In a real app, this would extract actual features
        # For demo, just return random features
        return [random.random() for _ in range(128)]

    def train_model(self):
        # In a real app, this would train an XGBoost model
        self.trained = True
        return True

    def recognize(self, img_data):
        # In a real app, this would do actual recognition
        # For demo, just return a random user if we have users
        if not users:
            return None, 0

        user = random.choice(users)
        confidence = random.uniform(0.7, 0.99)
        return user['id'], confidence

# Initialize face recognizer
face_recognizer = MockFaceRecognizer()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    global next_user_id

    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        image_data = request.form.get('image_data')

        if not name or not email or not image_data:
            flash('All fields are required', 'danger')
            return redirect(url_for('register'))

        # Check if user already exists
        if any(user['email'] == email for user in users):
            flash('Email already registered', 'danger')
            return redirect(url_for('register'))

        try:
            # Process the image data
            image_data_parts = image_data.split(',')
            if len(image_data_parts) > 1:
                image_data = image_data_parts[1]

            # Extract face features (mock)
            face_features = face_recognizer.extract_features(image_data)

            # Save user to "database"
            user_id = next_user_id
            next_user_id += 1

            new_user = {
                'id': user_id,
                'name': name,
                'email': email,
                'registration_date': datetime.now(),
                'face_features': face_features
            }

            users.append(new_user)

            # Save the image
            try:
                image_binary = base64.b64decode(image_data)
                filename = f"user_{user_id}.jpg"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

                with open(filepath, 'wb') as f:
                    f.write(image_binary)
            except Exception as e:
                print(f"Error saving image: {str(e)}")

            # Train the model with the new data
            face_recognizer.train_model()

            flash('Registration successful!', 'success')
            return redirect(url_for('index'))

        except Exception as e:
            flash(f'Error during registration: {str(e)}', 'danger')
            return redirect(url_for('register'))

    return render_template('register.html')

@app.route('/recognize', methods=['GET', 'POST'])
def recognize():
    if request.method == 'POST':
        image_data = request.form.get('image_data')

        if not image_data:
            return jsonify({'status': 'error', 'message': 'No image data provided'})

        try:
            # Process the image data
            image_data_parts = image_data.split(',')
            if len(image_data_parts) > 1:
                image_data = image_data_parts[1]

            # Recognize face (mock)
            user_id, confidence = face_recognizer.recognize(image_data)

            if user_id is None:
                return jsonify({'status': 'error', 'message': 'No face detected or recognized'})

            # Get user details
            user = next((u for u in users if u['id'] == user_id), None)

            if not user:
                return jsonify({'status': 'error', 'message': 'User not found'})

            return jsonify({
                'status': 'success',
                'user_id': user_id,
                'name': user['name'],
                'email': user['email'],
                'confidence': float(confidence)
            })

        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)})

    return render_template('recognize.html')

@app.route('/users')
def users_list():
    return render_template('users.html', users=users)

@app.route('/retrain')
def retrain():
    try:
        face_recognizer.train_model()
        flash('Model retrained successfully', 'success')
    except Exception as e:
        flash(f'Error retraining model: {str(e)}', 'danger')
    return redirect(url_for('index'))

if __name__ == '__main__':
    # Add some sample users for demo
    if not users:
        users.append({
            'id': next_user_id,
            'name': 'Demo User',
            'email': 'demo@example.com',
            'registration_date': datetime.now(),
            'face_features': face_recognizer.extract_features(None)
        })
        next_user_id += 1

    app.run(debug=True)
