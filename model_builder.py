# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 11:06:21 2019

@author: shwet
"""
import sys
from pyspark import SparkContext
from pyspark.mllib.feature import HashingTF
from pyspark.mllib.regression import LabeledPoint
from pyspark.ml.classification import LogisticRegression
from pyspark.mllib.classification import LogisticRegressionWithLBFGS
from pyspark.sql.functions import split
if __name__ == "__main__":
#
    print ("This is the name of the script: ", sys.argv[0])
    print ("Number of arguments: ", len(sys.argv))
    print ("The arguments are: " , str(sys.argv))
#
    if len(sys.argv) != 3:
        print("Usage: spam ,non-spam file <file>", file=sys.stderr)
        exit(-1)

#   DEFINE your input path
    spampath = sys.argv[1]
    non_spampath = sys.argv[2]
    sc= SparkContext("local", "first app")

# Load 2 types of emails from text files: spam and nonspam (non-spam).
    spam = sc.textFile(spampath)
    nonspam = sc.textFile(non_spampath)
        
#// Create a HashingTF instance to map email text to vectors of 1024 features
    tf = HashingTF(numFeatures = 1024)
    spamFeatures = spam.map(lambda email: tf.transform(email.split(" ")))
    nonspamFeatures = nonspam.map(lambda email: tf.transform(email.split(" ")))
#   create training data for spam 
    spamExamples = spamFeatures.map(lambda features: LabeledPoint(1, features))
#   create training data for non-spam 
    nonspamExamples = nonspamFeatures.map(lambda features: LabeledPoint(0, features))
#Combining both the trained data
    training_data = spamExamples.union(nonspamExamples)
    training_data.cache() # Cache data since Logistic Regression is an iterative algorithm.
# Creating the model based on the training data
    model =  LogisticRegressionWithLBFGS.train(training_data)

#Saving the model
    model.save(sc,r"C:\Users\shwet\OneDrive\Documents\BigData\LR")
  