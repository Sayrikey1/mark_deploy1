from flask import Flask , render_template, request

import marks as m


app = Flask(__name__)

@app.route("/", methods = ['GET','POST'])
def marks():
    global mks
    mks = 0
    if request.method == "POST":
        hrs = request.form['hrs']
        mks = m.marks_prediction(hrs)
    else:
        print(0)
    return render_template("index.html", my_marks = mks)

# @app.route("/sub", methods = ["POST"])
# def submit():
#     # HTML --> .py
#     if request.method == 'POST':
#         name = request.form["username"]
#         #.py --> HTML
#         return render_template("sub.html", n = name)

if __name__ == "__main__":
    app.run(debug=True)