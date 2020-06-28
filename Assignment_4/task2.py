import sys
import findspark
findspark.init()
import itertools
import time
from pyspark import SparkContext, SparkConf


def write_results(detected_communities, output_file):
    with open(output_file, 'w') as result_file:
        result_file.write('\n'.join(["'" + "', '".join(community) for community in detected_communities]))
    result_file.close()


if __name__ == '__main__':
    start_time = time.time()

    argv = sys.argv
    conf = SparkConf()
    # conf.set("spark.driver.memory", "4g")
    # conf.set("spark.executor.memory", "4g")
    conf.setMaster('local[8]')
    conf.setAppName('Assignment_4')
    sc = SparkContext.getOrCreate(conf)
    sc.setLogLevel("ERROR")

    filter_threshold = int(argv[1])

    # load the data and drop the header
    data = sc.textFile(argv[2]).map(lambda x: x.split(',')).map(lambda x: (x[0], x[1]))
    header = data.first()
    user_business_pairs = data.filter(lambda x: x != header)

    # get user pairs that clear the threshold
    user_pairs = user_business_pairs\
        .map(lambda x: (x[1], x[0]))\
        .groupByKey()\
        .mapValues(lambda x: sorted(list(x)))\
        .flatMap(lambda x: [((pair[0], pair[1]), x[0]) for pair in itertools.combinations(x[1], 2)])\
        .groupByKey()\
        .mapValues(lambda x: list(set(x)))\
        .filter(lambda x: len(x[1]) >= filter_threshold)\
        .map(lambda x: x[0])

    user_pairs_collection = user_pairs.collect()

    # create vertex rdd
    nodes = user_pairs\
        .flatMap(lambda x: list(x)) \
        .distinct()
    nodes_collection = nodes.collect()

    # get all edges
    edges = user_pairs.union(user_pairs.map(lambda x: (x[1], x[0])))
    edges_dict = edges.collect()

    adjacency_list = edges.groupByKey().mapValues(lambda x: list(set(x)))

    adjacency_dict = adjacency_list.collectAsMap()

    # calculate betweeness score for each edge
    edge_betweeness_scores = nodes\
        .flatMap(lambda x: calculate_betweenness(x, adjacency_dict))\
        .reduceByKey(lambda a, b: a + b)\
        .map(lambda x: (x[0], x[1]/2))\
        .sortBy(lambda x: (-x[1], x[0]))

    edge_betweeness_scores_collection = edge_betweeness_scores.collect()


    print('Duration: {:.2f}'.format(time.time() - start_time))

    print('completed')



















