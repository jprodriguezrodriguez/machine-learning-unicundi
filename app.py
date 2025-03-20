import re
from flask import Flask, render_template, request
from datetime import datetime

import linearRegressionML as lr

#import LinearRegressionExcelML|

app = Flask(__name__)

@app.route("/")
def home():
    return "Hello, Flask"

@app.route("/hello/<name>")
def hello_there(name):
    now =datetime.now()

    match_object = re.fullmatch("[a-zA-Z]+",name)
    if match_object:
        clean_name = match_object.group(0)
    else:
        clean_name = "Friend"
    content = f"Hello there  {clean_name} ! Hour:  {now}"
    return content
@app.route("/example")
def exampleHTML():
    return render_template("example.html")

@app.route("/linearregressionpage", methods=["GET", "POST"])
def calculateGrade():
    calculateResult = None
    plot_url = None
    hours = None

    if request.method == "POST":
        try:
            hours = float(request.form["Hours"])
            calculateResult = lr.calculateGrade(hours)
            plot_url = lr.generate_plot(hours)  # Generar gráfica solo si el input es válido
        except ValueError:
            calculateResult = "Invalid input. Please enter a number."
            plot_url = None  # Evita mostrar una gráfica si hay error en la entrada
    
    return render_template("linearRegressionGrades.html", result=calculateResult, plot_url=plot_url, hours=hours)

if __name__ == "__main__":
    app.run(debug=True)
@app.route("/mindmeister")
def MapaMental():
    return render_template("MapaMental.html")
