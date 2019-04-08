


import sys
from pyspark import SparkContext
from pyspark.mllib.feature import HashingTF
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import LogisticRegressionWithLBFGS, LogisticRegressionModel
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


    sc = SparkContext(appName="EmailSpamDetection")

    # Load 2 types of emails from text files: spam and nonspam (non-spam).
    # Each line has text from one email.

    model_path =sys.argv[1]
    querydata = sc.textFile(sys.argv[2])
    #Loading the saved model 
    savedModel = LogisticRegressionModel.load(sc, model_path)
    tf = HashingTF(numFeatures = 1024)
    #taking new emails predicting based on the trained model
    query = querydata.map(lambda emailids:(emailids.split(":")[0], savedModel.predict((tf.transform(emailids.split(":")[1].split(" "))))))
    output =  query.collect()
    for (classification, emailid) in output:
        print("%i: %s" % (classification, emailid))




