import sys
import findspark
from pyspark import SparkContext
import json

findspark.init()


def hash_partition_business_id(id):
    # return sum(list(map(ord, id))) % argv[4]
    return ord(id[0]) % argv[4]


if __name__ == '__main__':
    argv = sys.argv
    argv[4], argv[5] = list(map(int, argv[4:]))
    results = dict()
    sc = SparkContext('local[8]', 'task2')
    reviews_count_tuples = sc.textFile(argv[1]).map(lambda x: json.loads(x)).map(lambda x: (x['business_id'], 1))

    if argv[3] == 'default':
        business_review_map = reviews_count_tuples
        # business_review_map = reviews_data.map(lambda x: (x['business_id'], 1))

    else:
        business_review_map = reviews_count_tuples.partitionBy(argv[4], hash_partition_business_id)

    # business_review_map = reviews_data.map(lambda x: (x[0], 1))
    business_review_count = business_review_map.reduceByKey(lambda a, b: a + b)

    results['n_partitions'] = business_review_map.getNumPartitions()
    results['n_items'] = business_review_map.glom().map(len).collect()
    results['result'] = business_review_count.filter(lambda x: x[1] > argv[5]).map(lambda x: list(x)).collect()

    with open(argv[2], 'w') as result_file:
        result_file.write(json.dumps(results))
        result_file.close()





