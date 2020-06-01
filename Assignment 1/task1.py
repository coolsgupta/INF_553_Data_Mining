# essential to initialize
import json
import findspark
findspark.init()

from pyspark import SparkContext
sc = SparkContext('local', 'First App')

# load reveiws into RDD
reviews_txt = sc.textFile('asnlib/publicdata/review.json')

reviews_json = reviews_txt.map(lambda x: json.loads(x))

# 1.A
reviews_json.count()

# 1.B
reviews_json.filter(lambda x: '2012' in x['date']).count()


reviews_json.filter(lambda x: '2018' in x['date']).count()


reviews_json.map(lambda x: (x['date'][0:4], 1)).reduceByKey(lambda a,b: a+b).sortByKey(ascending=False).collect()

# 1.C
reviews_json.map(lambda x: x['user_id']).distinct().count()

# 1.D
user_review_count = reviews_json.map(lambda x: (x["user_id"], 1)).reduceByKey(lambda a, b: a + b)

user_review_count_sorted = user_review_count.sortBy(lambda x: x[1], ascending=False)

user_review_count_sorted.take(5)



