from app import db
import json

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    registration_date = db.Column(db.DateTime, nullable=False)
    face_data = db.relationship('FaceData', backref='user', lazy=True, uselist=False)
    
    def __repr__(self):
        return f'<User {self.name}>'

class FaceData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    _face_features = db.Column(db.Text, nullable=False)
    
    @property
    def face_features(self):
        return json.loads(self._face_features)
    
    @face_features.setter
    def face_features(self, value):
        self._face_features = json.dumps(value)
    
    def __repr__(self):
        return f'<FaceData for user_id {self.user_id}>'
