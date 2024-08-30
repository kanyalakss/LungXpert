from flask import Flask, render_template, request, url_for, redirect, session, make_response
from pymongo import MongoClient
import bcrypt
import os
from upload import upload_bp

# set app as a Flask instance 
app = Flask(__name__)
# encryption relies on secret keys so they could be run
app.secret_key = "testing"
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
app.static_folder = 'static'
# ลงทะเบียน Blueprint
app.register_blueprint(upload_bp)
# connect to your Mongo DB database
# def MongoDB():
#     client = MongoClient("mongodb+srv://pangkanya2545:<db_password>@cluster0.lmj1jbb.mongodb.net/")
#     db = client.get_database('ClassifyCXR')
#     records = db.register
#     return records

def localMongoDB():
    client = MongoClient("mongodb://localhost:27017/")
    db = client["ClassifyCXR"]
    records = db["register"]
    return records

records = localMongoDB()

@app.route("/", methods=['post', 'get'])
def index():
    if "email" in session:
        session.pop("email", None)
        return redirect(url_for("index"))
    response = make_response(render_template('index.html'))
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.route("/register", methods=['post', 'get'])
def register():
    if "email" in session:
        return redirect(url_for("logged_in"))
    
    message = ''
    if request.method == "POST":
        user = request.form.get("name")
        email = request.form.get("email")
        password1 = request.form.get("password")
        password2 = request.form.get("confirm_password")
        
        user_found = records.find_one({"name": user})
        email_found = records.find_one({"email": email})
        
        if user_found:
            message = 'There already is a user by that name'
        elif email_found:
            message = 'This email already exists in database'
        elif password1 != password2:
            message = 'Passwords should match!'
        else:
            hashed = bcrypt.hashpw(password2.encode('utf-8'), bcrypt.gensalt())
            user_input = {'name': user, 'email': email, 'password': hashed}
            records.insert_one(user_input)
            
            session["email"] = email
            return redirect(url_for("logged_in"))
    
    response = make_response(render_template('register.html', message=message))
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

@app.route("/login", methods=["POST", "GET"])
def login():
    if "email" in session:
        return redirect(url_for("logged_in"))
    
    message = 'Please login to your account'
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")

        email_found = records.find_one({"email": email})
        if email_found:
            email_val = email_found['email']
            passwordcheck = email_found['password']
            if bcrypt.checkpw(password.encode('utf-8'), passwordcheck):
                session["email"] = email_val
                return redirect(url_for('logged_in'))
            else:
                if "email" in session:
                    session.pop("email", None)
                message = 'Wrong password'
        else:
            message = 'Email not found'
    
    response = make_response(render_template('login.html', message=message))
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

@app.route("/logout", methods=["POST", "GET"])
def logout():
    if "email" in session:
        session.pop("email", None)
    response = make_response(redirect(url_for("index")))
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

@app.route('/logged_in')
def logged_in():
    if "email" not in session:
        session.pop("email", None)
        return redirect(url_for("login"))
    
    # ดึง email จาก session
    email = session['email']
    user = records.find_one({"email": email})
    name = user['name'] if user else "Guest"  # ถ้าไม่พบผู้ใช้จะแสดงเป็น "Guest"
    
    # ส่งข้อมูลทั้ง email และ name ไปยังเทมเพลต HTML
    response = make_response(render_template('logged_in.html', email=email, name=name))
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response



if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)