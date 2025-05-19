import cv2
import numpy as np
import xgboost as xgb
import os
import pickle
from sklearn.preprocessing import LabelEncoder

class FaceRecognizer:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.model = None
        self.label_encoder = LabelEncoder()
        self.model_path = 'face_recognition_model.pkl'
        
        # Load model if it exists
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    model_data = pickle.load(f)
                    self.model = model_data['model']
                    self.label_encoder = model_data['label_encoder']
            except Exception as e:
                print(f"Error loading model: {e}")
    
    def extract_features(self, image):
        """Extract features from a face image"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            return None
        
        # Take the first face (assuming one face per image)
        x, y, w, h = faces[0]
        
        # Extract face ROI
        face_roi = gray[y:y+h, x:x+w]
        
        # Resize to a fixed size
        face_roi = cv2.resize(face_roi, (100, 100))
        
        # Apply some preprocessing
        face_roi = cv2.equalizeHist(face_roi)
        
        # Extract HOG features
        hog = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)
        h = hog.compute(face_roi)
        
        # Flatten the features
        features = h.flatten()
        
        return features
    
    def train_model(self):
        """Train the XGBoost model with the face data from the database"""
        from app import db
        from models import User, FaceData
        
        # Get all face data
        face_data = FaceData.query.all()
        
        if not face_data:
            print("No face data available for training")
            return
        
        # Prepare training data
        X = []
        y = []
        
        for data in face_data:
            X.append(data.face_features)
            y.append(data.user_id)
        
        X = np.array(X)
        y = np.array(y)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Train XGBoost model
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            objective='multi:softprob',
            num_class=len(set(y_encoded))
        )
        
        self.model.fit(X, y_encoded)
        
        # Save the model
        with open(self.model_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'label_encoder': self.label_encoder
            }, f)
        
        print("Model trained and saved successfully")
    
    def recognize(self, image):
        """Recognize a face in the image"""
        if self.model is None:
            print("Model not trained yet")
            return None, 0
        
        # Extract features
        features = self.extract_features(image)
        
        if features is None:
            return None, 0
        
        # Reshape for prediction
        features = features.reshape(1, -1)
        
        # Predict
        probabilities = self.model.predict_proba(features)[0]
        predicted_class_idx = np.argmax(probabilities)
        confidence = probabilities[predicted_class_idx]
        
        # Get the original user ID
        predicted_user_id = self.label_encoder.inverse_transform([predicted_class_idx])[0]
        
        # Only return a match if confidence is high enough
        if confidence < 0.6:  # Threshold can be adjusted
            return None, 0
        
        return predicted_user_id, confidence
