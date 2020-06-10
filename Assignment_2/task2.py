import time
import itertools
import findspark
findspark.init()
from pyspark import SparkContext, SparkConf, StorageLevel
import sys

def get_all_candidates(bucket, support, total_baskets_count):
    all_candidates = []
    singleton_frequency_map = {}
    candidate_singletons = []
    pair_frequency_map = {}
    candidate_pairs = []
    partition_support = round(support * (len(bucket) / total_baskets_count))

    for basket in bucket:
        basket.sort()
        for item in basket:
            if item not in singleton_frequency_map:
                singleton_frequency_map[item] = 0
            singleton_frequency_map[item] += 1

        for pair in itertools.combinations(basket, 2):
            if frozenset(pair) not in pair_frequency_map:
                pair_frequency_map[frozenset(pair)] = 0
            pair_frequency_map[frozenset(pair)] += 1

    for item in singleton_frequency_map:
        if singleton_frequency_map[item] >= partition_support:
            candidate_singletons.append(item)

    all_candidates.append([1, candidate_singletons])

    for pair in itertools.combinations(candidate_singletons, 2):
        pair = frozenset(pair)
        if pair in pair_frequency_map and pair_frequency_map[pair] >= partition_support:
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

                    if part_sup_cand >= partition_support and cand_item_set not in supported_candidates:
                        supported_candidates.append(cand_item_set)

        # eliminate_candidates = []
        verified_candidates = []
        for cand_item_set in supported_candidates:
            valid_candidate = True
            for cand_subset in itertools.combinations(cand_item_set, set_size - 1):
                if set(cand_subset) not in all_candidates[set_size - 2][1]:
                    valid_candidate = False
                    # eliminate_candidates.append(subset)

            if valid_candidate and cand_item_set not in verified_candidates:
                verified_candidates.append(cand_item_set)


        # supported_candidates = list(set(supported_candidates) - set(eliminate_candidates))

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


def get_original_itemset_counts(basket, all_candidates):
    for candidate_set in all_candidates:
        for candidate in candidate_set:
            # for basket in bucket:
            if set(candidate).issubset(basket):
                yield (candidate, 1)


def get_candidates_list(all_baskets, total_baskets_count):
    all_candidates = all_baskets \
        .mapPartitions(lambda x: get_all_candidates(list(x), support, total_baskets_count)) \
        .reduceByKey(lambda a, b: a + b) \
        .map(lambda x: list(set(x[1]))) \
        .sortBy(lambda x: len(x[0])) \
        .collect()

    return all_candidates


def get_original_frequent_sets(original_baskets, candidate_list):
    frequent_itemset_list = original_baskets\
        .flatMap(lambda x: get_original_itemset_counts(x, candidate_list))\
        .reduceByKey(lambda a, b: a+b)\
        .filter(lambda x: x[1] >= support)\
        .map(lambda x: (len(x[0]), x[0]))\
        .groupByKey()\
        .sortBy(lambda x: x[0])\
        .map(lambda x: list(x[1]))\
        .collect()

    return frequent_itemset_list


def write_results(result_candidates, result_frequent_itemsets, result_file_path):
    with open(result_file_path, 'w') as results_file:
        results_file.write('Candidates:\n')
        output = []
        for single_cad in sorted(result_candidates[0]):
            output.append('(\'' + str(single_cad[0]) + '\')')

        results_file.write(','.join(output) + '\n\n')

        for cand_set in result_candidates[1:]:
            results_file.write(','.join(map(str, (sorted(cand_set)))) + '\n\n')

        results_file.write('Frequent Itemsets:\n')
        output = []
        for single_item in sorted(result_frequent_itemsets[0]):
            output.append('(\'' + str(single_item[0]) + '\')')

        results_file.write(','.join(output) + '\n\n')

        for freq_set in result_frequent_itemsets[1:]:
            results_file.write(','.join(map(str, (sorted(freq_set)))) + '\n\n')


if __name__ == '__main__':
    start_time = time.time()

    # initialize spark
    conf = SparkConf()
    conf.set("spark.driver.memory", "4g")
    conf.set("spark.executor.memory", "4g")
    conf.setMaster('local[8]')
    conf.setAppName('Assignment_2')
    sc = SparkContext.getOrCreate(conf)

    # get args
    threshold = int(sys.argv[1])
    support = int(sys.argv[2])
    input_file = sys.argv[3]
    result_file = sys.argv[4]

    # create baskets rdd
    data = sc.textFile(input_file).map(lambda x: x.split(',')).map(lambda x: (x[0], x[1]))
    header = data.first()
    raw_data = data.filter(lambda x: x != header)

    baskets = raw_data.groupByKey().map(lambda x: (list(set(x[1])))).filter(lambda x: len(x) >= threshold)
    baskets = sc.parallelize(baskets.collect(), 2)
    total_baskets_count = baskets.count()

    # initialize partial_support
    # partition_support = round(support / baskets.getNumPartitions())

    # get candidate frequent item sets
    candidate_item_sets = get_candidates_list(baskets, total_baskets_count)

    # evaluate candidate frequent item sets for original frequent item sets
    frequent_itemsets = get_original_frequent_sets(
        original_baskets=baskets,
        candidate_list=candidate_item_sets
    )

    # write results to file
    write_results(
        result_candidates=candidate_item_sets,
        result_frequent_itemsets=frequent_itemsets,
        result_file_path=result_file
    )

    # output 1
    print("Duration: {:.2f}".format(time.time() - start_time))