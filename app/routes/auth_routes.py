from flask import Blueprint, request, jsonify, session
from ..components.FaceRegistration import FaceGestureRegistration
from ..models.user import User
from .. import db

@auth_bp.route('/register', methods=['POST'])
def register():
    try:
        data = request.get_json()
        username = data.get('username')
        email = data.get('email')
        password = data.get('password')
        
        if not all([username, email, password]):
            return jsonify({'error': 'All fields are required'}), 400
        
        # Check if user already exists
        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            return jsonify({'error': 'Email already registered'}), 400
        
        # Create new user
        new_user = User(username=username, email=email)
        new_user.set_password(password)
        
        db.session.add(new_user)
        db.session.commit()
        
        # Store user_id in session for face registration
        session['registration_user_id'] = new_user.id
        
        return jsonify({
            'status': 'success',
            'message': 'Basic registration successful. Proceed to face and gesture registration.',
            'user_id': new_user.id
        })

    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@auth_bp.route('/start_face_registration', methods=['POST'])
def start_face_registration():
    try:
        user_id = session.get('registration_user_id')
        if not user_id:
            return jsonify({'error': 'No active registration session'}), 400

        face_gesture_reg = FaceGestureRegistration()
        registration_success = face_gesture_reg.register_user(user_id)

        if registration_success:
            # Update user status in database
            user = User.query.get(user_id)
            if user:
                user.face_registered = True
                user.registration_complete = True
                db.session.commit()
                
                # Clear registration session
                session.pop('registration_user_id', None)
                
                return jsonify({
                    'status': 'success',
                    'message': 'Face and gesture registration completed successfully'
                })
        
        return jsonify({
            'status': 'error',
            'message': 'Registration was not completed'
        }), 400

    except Exception as e:
        db.session.rollback()
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500 