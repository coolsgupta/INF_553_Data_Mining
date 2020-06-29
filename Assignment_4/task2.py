import sys
import findspark

findspark.init()
import itertools
import time
from pyspark import SparkContext, SparkConf
import random

def write_results(result, output_file):
    with open(output_file, 'w') as result_file:
        result_file.write('\n'.join([str(edge_score)[1:-1] for edge_score in result]))
    result_file.close()


def compute_bfs_traversal_from_node(root_node, adjacency_map_dict):
    traversal_depth_child_map = {
        root_node: {
            'depth': 0,
            'parent': []
        }
    }
    visited_nodes_set = set()
    traversal_stack = [root_node]

    while (traversal_stack):
        parent = traversal_stack.pop(0)
        visited_nodes_set.add(parent)
        for child in adjacency_map_dict.get(parent):
            if child not in visited_nodes_set:
                visited_nodes_set.add(child)
                traversal_depth_child_map[child] = {
                    'depth': traversal_depth_child_map[parent]['depth'] + 1,
                    'parent': [parent]
                }
                traversal_stack.append(child)

            elif traversal_depth_child_map[child]['depth'] == traversal_depth_child_map[parent]['depth'] + 1:
                traversal_depth_child_map[child]['parent'].append(parent)

    return traversal_depth_child_map


def compute_edge_weights(traversal_depth_child_map):
    edge_weight_map = dict.fromkeys(list(traversal_depth_child_map.keys()), 1)
    tree_struct = {}
    for node in traversal_depth_child_map:
        if traversal_depth_child_map[node]['depth'] not in tree_struct:
            tree_struct[traversal_depth_child_map[node]['depth']] = []
        tree_struct[traversal_depth_child_map[node]['depth']].append(tuple([node, traversal_depth_child_map[node]['parent']]))

    path_count_map = {}
    for depth in sorted(tree_struct.keys()):
        for child, parents in tree_struct[depth]:
            path_count_map[child] = sum(
                [path_count_map[immediate_parent] for immediate_parent in parents]
            ) if parents else 1

    node_parent_total_weight = {}
    for node in traversal_depth_child_map:
        if traversal_depth_child_map[node]['parent']:
            num_paths = sum([path_count_map[parent_node] for parent_node in traversal_depth_child_map[node]['parent']])
            for parent in traversal_depth_child_map[node]['parent']:
                propagated_weight = edge_weight_map[node]/num_paths
                total_node_weight = propagated_weight*path_count_map[parent]
                node_parent_total_weight[tuple(sorted([node, parent]))] = total_node_weight
                edge_weight_map[parent] += total_node_weight

    return node_parent_total_weight


def get_GN_betweeness_scores(**kwargs):
    edge_betweeness_scores_map = {}
    for root_node in kwargs.get('vertices'):
        bfs_traversal_map = {
            node: depth_parent_pair for node, depth_parent_pair in sorted(
                compute_bfs_traversal_from_node(root_node, kwargs.get('adjacency_map_dict')).items(),
                key=lambda x: -x[1]['depth']
            )
        }
        edge_weight_map = compute_edge_weights(bfs_traversal_map)
        for edge, score in edge_weight_map.items():
            if edge not in edge_betweeness_scores_map:
                edge_betweeness_scores_map[edge] = 0
            edge_betweeness_scores_map[edge] += edge_weight_map[edge]

    edge_betweeness_scores_sorted_list = sorted(
        [(edge, score/2) for edge, score in edge_betweeness_scores_map.items()],
        key=lambda x: (-x[1], x[0][0])
    )

    return edge_betweeness_scores_sorted_list


def get_current_communities(**kwargs):
    vertices = kwargs.get('vertices')
    adjacency_map_dict = kwargs['adjacency_map_dict']

    communities = list()  # result will be return
    need2visit = list()  # a stack actually
    temp_node_set = set()  # using to save each communities
    visited = set()  # track which node has been visited

    # random pick a root to detect communities
    random_root = vertices[random.randint(0, len(vertices) - 1)]
    temp_node_set.add(random_root)
    need2visit.append(random_root)
    # if still has some node we haven't visit, do the loop
    while len(visited) != len(vertices):
        while len(need2visit) > 0:
            parent_node = need2visit.pop(0)
            temp_node_set.add(parent_node)
            visited.add(parent_node)
            for children in adjacency_map_dict[parent_node]:
                if children not in visited:
                    temp_node_set.add(children)
                    need2visit.append(children)
                    visited.add(children)

        communities.append(sorted(temp_node_set))
        temp_node_set = set()
        if len(vertices) > len(visited):
            # pick one from rest of unvisited nodes
            need2visit.append(set(vertices).difference(visited).pop())

    return communities


def find_communities(**kwargs):
    vertices = kwargs.get('vertices')
    edge_betweenness_scores = kwargs['edge_betweenness_scores']
    adjacency_map_dict = kwargs['adjacency_map_dict']
    edges = kwargs['edges']
    modularity = float("-inf")
    communities = []

    while True:
        if edge_betweenness_scores:
            removed_edge = edge_betweenness_scores.pop(0)[0]
            try:
                adjacency_map[removed_edge[0]].remove(removed_edge[1])

            except:
                pass

            try:
                adjacency_map[removed_edge[1]].remove(removed_edge[0])

            except:
                pass

            current_communities = get_current_communities(
                vertices=vertices,
                adjacency_map_dict=adjacency_map_dict
            )

            sum_mod = 0
            for com in current_communities:
                sum_mod += sum(
                    [
                        (1 if tuple(sorted([pair[0], pair[1]])) in edges else 0) -
                        (len(adjacency_map_dict[pair[0]]) * len(adjacency_map_dict[pair[1]]) / (2 * len(edges)))
                        for pair in itertools.combinations(list(com), 2)
                    ]
                )

            current_modularity = sum_mod / (2 * len(edges))

            if current_modularity < modularity:
                break

            communities = current_communities
            modularity = current_modularity

    return sorted(
        communities,
        key=lambda item: (len(item), item[0], item[1])
    )



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
    user_pairs = user_business_pairs \
        .map(lambda x: (x[1], x[0])) \
        .groupByKey() \
        .mapValues(lambda x: sorted(list(x))) \
        .flatMap(lambda x: [((pair[0], pair[1]), x[0]) for pair in itertools.combinations(x[1], 2)]) \
        .groupByKey() \
        .mapValues(lambda x: list(set(x))) \
        .filter(lambda x: len(x[1]) >= filter_threshold) \
        .map(lambda x: x[0])

    user_pairs_collection = user_pairs.collect()

    # create vertex rdd
    nodes = user_pairs \
        .flatMap(lambda x: list(x)) \
        .distinct()
    nodes_collection = sorted(nodes.collect())

    # get all edges
    # edges = user_pairs.union(user_pairs.map(lambda x: (x[1], x[0])))
    # edges_collection = edges.collect()

    adjacency_map = user_pairs \
        .union(user_pairs.map(lambda x: (x[1], x[0]))) \
        .groupByKey() \
        .mapValues(lambda x: list(set(x)))

    adjacency_dict = adjacency_map.collectAsMap()

    # get betweeness score for edges
    edge_betweenness_scores_list = get_GN_betweeness_scores(
        vertices=nodes_collection,
        adjacency_map_dict=adjacency_dict
    )

    write_results(edge_betweenness_scores_list, argv[3])

    detected_communities = find_communities(
        vertices=nodes_collection,
        edge_betweenness_scores=edge_betweenness_scores_list,
        adjacency_map_dict=adjacency_dict,
        edges=user_pairs_collection
    )

    write_results(detected_communities, argv[4])



    print('Duration: {:.2f}'.format(time.time() - start_time))

    print('completed')

    # adjacency_map = edges
    # user_pairs = A_matrix
    # node_weight_map = vertex_weight_map
    # nodes_collection = vertexes
    # original_edges = copy of initial adjacency_map
    # m = 498 number of edges
