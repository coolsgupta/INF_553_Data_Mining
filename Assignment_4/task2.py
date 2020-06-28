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


def calculate_betweenness(root_node, adjacent_vertices):
    traversal_order = [root_node]
    nodes_visited = [root_node]
    node_depth_map = {root_node: 0}
    node_parent_map = {}
    node_weight_map = {root_node: 1}

    while (traversal_order):
        node = traversal_order.pop(0)
        children = adjacent_vertices[node]

        for i in children:
            if (i not in nodes_visited):
                traversal_order.append(i)
                node_parent_map[i] = [node]
                node_weight_map[i] = node_weight_map[node]
                nodes_visited.append(i)
                node_depth_map[i] = node_depth_map[node] + 1

            else:
                if (i != root_node):
                    node_parent_map[i].append(node)
                    if (node_depth_map[node] == node_depth_map[i] - 1):
                        node_weight_map[i] += node_weight_map[node]

    order_v = []
    count = 0
    for i in nodes_visited:
        order_v.append((i, count))
        count = count + 1
    reverse_order = sorted(order_v, key=(lambda x: x[1]), reverse=True)
    rev_order = []
    nodes_values = {}
    for i in reverse_order:
        rev_order.append(i[0])
        nodes_values[i[0]] = 1

    betweenness_values = {}

    for j in rev_order:
        if (j != root_node):
            total_weight = 0
            for i in node_parent_map[j]:
                if (node_depth_map[i] == node_depth_map[j] - 1):
                    total_weight += node_weight_map[i]

            for i in node_parent_map[j]:
                if (node_depth_map[i] == node_depth_map[j] - 1):
                    source = j
                    dest = i

                    if source < dest:
                        pair = tuple((source, dest))
                    else:
                        pair = tuple((dest, source))

                    if (pair not in betweenness_values.keys()):
                        betweenness_values[pair] = float(nodes_values[source] * node_weight_map[dest] / total_weight)
                    else:
                        betweenness_values[pair] += float(nodes_values[source] * node_weight_map[dest] / total_weight)

                    nodes_values[dest] += float(nodes_values[source] * node_weight_map[dest] / total_weight)

    betweenness_list = []
    for key, value in betweenness_values.items():
        temp = [key, value]
        betweenness_list.append(temp)

    return betweenness_list


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

    print('Duration: {:.2f}'.format(time.time() - start_time))

    print('completed')



















