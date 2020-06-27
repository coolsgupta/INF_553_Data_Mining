import os
import sys
import findspark
findspark.init()
os.environ["PYSPARK_SUBMIT_ARGS"] = "--packages graphframes:graphframes:0.6.0-spark2.3-s_2.11 pyspark-shell"
import itertools

import time
from pyspark import SparkContext, SparkConf
from graphframes import GraphFrame
from pyspark.sql import SparkSession


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
    conf.setMaster('local[16]')
    conf.setAppName('Assignment_4')
    sc = SparkContext.getOrCreate(conf)
    spark = SparkSession(sc)
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
        .map(lambda x: tuple([x]))\
        .distinct()

    nodes_df = nodes.toDF(["id"])
    # nodes_df.show()

    # get all possible edges
    edges_df = user_pairs.union(user_pairs.map(lambda x: (x[1],x[0]))).toDF(["src", "dst"])
    # edges_df.show()

    # create graph
    community_graph = GraphFrame(nodes_df, edges_df)

    # get labels
    result = community_graph.labelPropagation(maxIter=5)
    # result.show()

    # group communities
    detected_communities = result\
        .rdd\
        .map(tuple)\
        .map(lambda x: (x[1], x[0]))\
        .groupByKey()\
        .map(lambda x: sorted(list(x[1])))\
        .sortBy(lambda x: (len(x), x[0]))

    detected_communities_collection = detected_communities.collect()

    write_results(detected_communities_collection, argv[3])

    print('Duration: {:.2f}'.format(time.time() - start_time))

    print('completed')
