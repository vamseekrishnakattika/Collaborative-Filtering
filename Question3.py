from pyspark import SparkConf, SparkContext
from pyspark.mllib.recommendation import ALS
import math

conf = SparkConf().setMaster("local").setAppName('Question3')
sc = SparkContext(conf=conf)

ratings_data = sc.textFile("ratings.dat").map(lambda x: x.split("::"))

ratings_data = ratings_data.map(lambda x: (x[0], x[1], x[2]))

train_rdd, test_rdd = ratings_data.randomSplit([6, 4], seed=0)

test_for_predict = test_rdd.map(lambda x: (x[0], x[1]))

seed = 10
iterations = 20
ranks = [4, 8, 10, 12, 16]
errors = [0, 0, 0, 0, 0]
err = 0
reg = 0.1

min_error = float('inf')
best_rank = -1

for rank in ranks:
    model = ALS.train(train_rdd, rank, seed=seed, iterations=iterations, lambda_=reg , nonnegative=False, blocks=-1)
    predictions = model.predictAll(test_for_predict).map(lambda r: (((r[0], r[1]), r[2])))
    ratesAndPreds = test_rdd.map(lambda x: (((int(x[0]), int(x[1])), float(x[2])))).join(predictions)
    error = math.sqrt(ratesAndPreds.map(lambda x: (x[1][0] - x[1][1]) ** 2).mean())
    errors[err] = error
    err += 1

    print("For rank %s the RMSE is %s" % (rank, error))

    if (error < min_error):
        min_error = error
        best_rank = rank
print("The best model was trained with rank %s", best_rank)
