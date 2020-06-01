# essential to initialize
import json
import sys
import findspark

findspark.init()
from pyspark import SparkContext

if __name__ == '__main__':
    sc = SparkContext('local[8]', 'task_1')
    # sc = SparkContext()
    results = {}
    argv = sys.argv

    # load reveiws into RDD
    reviews_txt = sc.textFile(argv[1])

    reviews_json = reviews_txt.map(lambda x: json.loads(x))

    # 1.A
    results['A'] = reviews_json.count()

    # 1.B
    results['B'] = reviews_json.filter(lambda x: argv[4] in x['date']).count()

    # 1.C
    results['C'] = reviews_json.map(lambda x: x['user_id']).distinct().count()

    # 1.D
    user_review_count = reviews_json.map(lambda x: (x["user_id"], 1)).reduceByKey(lambda a, b: a + b)

    user_review_count_sorted = user_review_count.sortBy(lambda x: x[1], ascending=False)

    results['D'] = user_review_count_sorted.take(int(argv[5]))

    # 1.E
    words_count_rdd = reviews_json.map(lambda x: (x['text'])).flatMap(lambda line: line.lower().split()).map(
        lambda x: (x.strip(), 1)).reduceByKey(lambda a, b: a + b)
    with open(argv[3]) as f:
        f_stopwords = f.read()
        f.close()

    stopwords_list = [x.strip().lower() for x in f_stopwords.split()]
    stopwords_list.extend(["(", "[", ",", ".", "!", "?", ":", ";", "]", ")"])
    results['E'] = words_count_rdd.filter(lambda x: x[0] not in stopwords_list).sortBy(lambda x: x[1],
                                                                                       ascending=False).map(
        lambda x: x[0]).take(int(argv[5]))

    with open(argv[2], 'w') as results_file:
        results_file.write(json.dumps(results))


