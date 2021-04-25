#importing required libraries

from flask import Flask
from flask import render_template, request, url_for, redirect, jsonify
import pyspark
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.sql.types import IntegerType, DoubleType
from pyspark.ml.pipeline import PipelineModel
import sys
from errors import InvalidUsage
import json
import findspark
findspark.init()
  
app = Flask(__name__) #creating the Flask class object   

# creating a spark SparkSession
spark = SparkSession.builder \
    .master("local") \
    .appName("spark-app") \
    .getOrCreate()





@app.route('/')
def home():
    """
    home page for our app
    """
    return render_template('form_css.html')

@app.route('/churn')
def churn():
    """
    redirects here if the prediction is churn
    """
    return 'Caution! This customer will churn. Take necessary steps to retain him!'

@app.route('/nochurn')
def nochurn():
    """
    redirects here if the prediction is not churn
    """
    return 'No need to worry! This customer will not churn.'

@app.route('/unable')
def unable():
    """
    redirects here if the prediction if we are unable to process the input
    """
    return 'We are unable to process the input! Kindly enter again'

@app.errorhandler(InvalidUsage)
def handle_invalid_usage(error):
    """
    handling invalid data types from user
    """
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response


@app.route('/predict', methods=['GET','POST'])
def predict():

    """
    Model is loaded, Input from the user is
    processed into a Pyspark DataFrame and 
    the prediction is given
    """

    #reading user inputs
    gender = request.form['gender']
    avgEvents = request.form['avgEvents']
    avgSongs = request.form['avgSongs']
    thumbsup = request.form['thumbsup']
    thumbsdown = request.form['thumbsdown']
    add_friend = request.form['add_friend']
    reg_date = request.form['reg_date']
    level = request.form['level']

    def validate(date_text):

        """
        This function validates that the 
        input Data Type for all the fields 
        is correct. If not an error is thrown
        which is handled
        """
        try:
            datetime.strptime(date_text, '%Y-%m-%d')
            int(avgEvents)
            int(thumbsup)
            int(thumbsdown)
            int(add_friend)
            int(avgSongs)
        except :
            raise InvalidUsage("Incorrect data type for input parameters! Please retry", status_code=410)

    validate(reg_date)


    # Printing out to the console to validate that the input is read correctly
    # print('Hello world!', file=sys.stderr)
    # print('gender ',gender, file=sys.stderr)
    # print('avgEvents',avgEvents,file=sys.stderr)
    # print('avgSongs',avgSongs,file=sys.stderr)
    # print(reg_date, thumbsdown,thumbsup,level,file=sys.stderr)

    days_active = (datetime.now() - datetime.strptime(reg_date, '%Y-%m-%d')).days
    
    #creating a Spark Context to make a RDD out of input data and then convert to DataFrame
    sc = SparkContext.getOrCreate()

    df = sc.parallelize([[gender, level, days_active, avgSongs, avgEvents, thumbsup, thumbsdown, add_friend]]).toDF(["gender", "last_level", "days_active", "avg_songs", "avg_events", "thumbs_up", "thumbs_down", "addfriend"])

    df = df.withColumn("days_active", df["days_active"].cast(IntegerType()))
    df = df.withColumn("avg_songs", df["avg_songs"].cast(DoubleType()))
    df = df.withColumn("avg_events", df["avg_events"].cast(DoubleType()))
    df = df.withColumn("thumbs_up", df["thumbs_up"].cast(IntegerType()))
    df = df.withColumn("thumbs_down", df["thumbs_down"].cast(IntegerType()))
    df = df.withColumn("addfriend", df["addfriend"].cast(IntegerType()))

    model = PipelineModel.load(r'C:\Users\shubh\OneDrive\Desktop\notebooks\model-rf-with-onehotestimator')
    pred = model.transform(df)
    result = int(pred.select(pred.prediction).collect()[0].prediction)

    #If no prediction given
    if pred.count()==0:
        redirect(url_for('unable'))
    else:
    
        if result == 1:
            return redirect(url_for('churn'))
        elif result == 0:
            return redirect(url_for('nochurn'))

 
if __name__ =='__main__':  
    app.run() 
    
    

