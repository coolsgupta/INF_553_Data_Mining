import time
import itertools
import findspark
findspark.init()
from pyspark import SparkContext, SparkConf, StorageLevel
import sys

conf = SparkConf()
conf.set("spark.driver.memory", "4g")
conf.set("spark.executor.memory", "4g")
conf.setMaster('local[8]')
conf.setAppName('Assignment_2')
sc = SparkContext.getOrCreate(conf)


def get_all_candidates(bucket):
    all_candidates = []
    singleton_count_map = {}
    candidate_singletons = []
    pair_count_map = {}
    candidate_pairs = []

    for basket in bucket:
        basket.sort()
        for item in basket:
            if item not in singleton_count_map:
                singleton_count_map[item] = 0
            singleton_count_map[item] += 1

        for pair in itertools.combinations(basket, 2):
            if frozenset(pair) not in pair_count_map:
                pair_count_map[frozenset(pair)] = 0
            pair_count_map[frozenset(pair)] += 1

    for item in singleton_count_map:
        if singleton_count_map[item] >= partition_support:
            candidate_singletons.append((item))

    all_candidates.append([1, candidate_singletons])

    for pair in itertools.combinations(candidate_singletons, 2):
        pair = frozenset(pair)
        if pair in pair_count_map and pair_count_map[pair] >= partition_support:
            candidate_pairs.append(pair)

    all_candidates.append([2, candidate_pairs])

    set_size = 3
    previous_candidates = candidate_pairs

    while 1:
        current_candidates = []
        for i, subcand_1 in enumerate(previous_candidates):
            for subcand_2 in previous_candidates[i + 1:]:
                superset_cand = subcand_1.union(subcand_2)

                if len(superset_cand) == set_size and superset_cand not in current_candidates:
                    current_candidates.append(superset_cand)

        supported_candidates = []
        for cand_item_set in current_candidates:
            part_sup_cand = 0
            for basket in bucket:
                if cand_item_set.issubset(set(basket)):
                    part_sup_cand += 1

                    if part_sup_cand >= partition_support:
                        supported_candidates.append(cand_item_set)

        for cand_item_set in supported_candidates:
            for subset in itertools.combinations(cand_item_set, set_size - 1):
                subset = set(subset)
                if subset not in all_candidates[set_size - 2][1]:
                    supported_candidates.remove(cand_item_set)

        previous_candidates = supported_candidates
        set_size += 1

        if not previous_candidates:
            break
        all_candidates.append((set_size, supported_candidates))

    for cand_set in all_candidates:
        if cand_set[0] == 1:
            candidates = [tuple((x,)) for x in cand_set[1]]
        else:
            candidates = [tuple(sorted(x)) for x in cand_set[1]]
        yield ((cand_set[0], candidates))


def get_original_frequent_sets(bucket, all_candidates):
    for candidate_set in all_candidates:
        for candidate in candidate_set:
            for basket in bucket:
                if set(candidate).issubset(basket):
                    yield (candidate, 1)


def get_candidates_list(all_baskets):
    all_candidates = all_baskets \
        .mapPartitions(lambda x: get_all_candidates(list(x))) \
        .reduceByKey(lambda a, b: a + b) \
        .map(lambda x: list(set(x[1]))) \
        .sortBy(lambda x: len(x[0])) \
        .collect()

    return all_candidates


def get_original_frequent_sets(original_baskets, candidate_list):
    frequent_itemset_list = original_baskets\
        .mapPartitions(lambda x: get_original_frequent_sets(list(x), candidate_list))\
        .reduceByKey(lambda a, b: a+b)\
        .filter(lambda x: x[1] >= support)\
        .map(lambda x: (len(x[0]), [x[0]]))\
        .groupByKey()\
        .map(lambda x: (x[0], list(x[1])))\
        .sortBy(lambda x: x[0])\
        .collect()

    return frequent_itemset_list


if __name__ == '__main__':
    start_time = time.time()
    threshold = int(sys.argv[1])
    support = int(sys.argv[2])

    data = sc.textFile(sys.argv[3]).map(lambda x: x.split(',')).map(lambda x: (x[0], x[1]))
    header = data.first()
    raw_data = data.filter(lambda x: x != header)

    # if case == '2':
    #     raw_data = raw_data.map(lambda x: (x[1], x[0]))

    baskets = raw_data.groupByKey().map(lambda x: (list(set(x[1])))).filter(lambda x: len(x) >= threshold)
    # print(baskets.collect())
    partition_support = int(support / baskets.getNumPartitions())
    # candidate_item_sets = baskets\
    #     .mapPartitions(lambda x: get_all_candidates(list(x))) \
    #     .reduceByKey(lambda a, b: a + b)        \
    #     .map(lambda x: list(set(x[1])))\
    #     .filter(lambda x: len(x) > 0)\
    #     .sortBy(lambda x: len(x[0]))

    candidate_item_sets = get_candidates_list(baskets)

    frequent_itemsets = get_original_frequent_sets(
        original_baskets=baskets,
        candidate_list=candidate_item_sets
    )

    print(candidate_item_sets)