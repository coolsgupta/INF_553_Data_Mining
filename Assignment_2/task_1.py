import itertools
import findspark
findspark.init()
from pyspark import SparkContext, SparkConf
import sys
import time

def get_original_itemset_counts(basket):
    if set_size > 1:
        sets = itertools.combinations(basket[1], set_size)

    else:
        sets = basket[1]

    return list(map(lambda x: (x, 1), sets))


def get_partition_itemset_candidates(bucket):
    items = {}
    for basket in bucket:
        if len(basket[1]) < set_size:
            continue

        if set_size > 1:
            sets = itertools.combinations(basket[1], set_size)

        else:
            sets = basket[1]

        for item in sets:
            if item not in items:
                items[item] = 0

            items[item] += 1

    print(items)
    for item in items:
        if items[item] >= partial_support:
            yield item


def get_candidates_list(candidate_baskets):
    candidate_sets = candidate_baskets.mapPartitions(get_partition_itemset_candidates)
    candidate_sets_list = candidate_sets.distinct().collect()
    return candidate_sets_list


def get_original_frequent_sets(original_baskets, candidate_list):
    original_freq_sets = original_baskets \
        .flatMap(get_original_itemset_counts) \
        .filter(lambda x: x[0] in candidate_list) \
        .reduceByKey(lambda a, b: a + b) \
        .filter(lambda x: x[1] >= support)\
        .map(lambda x: (x[0]))\
        .collect()

    return original_freq_sets


def write_results(result_candidates, result_frequent_itemsets):
    with open(argv[4], 'w') as results_file:
        results_file.write('Candidates:\n')
        output = ''
        for single_cad in sorted(result_candidates[0]):
            output += '(\'' + str(single_cad) + '\'),'

        results_file.write(output[:-1] + '\n\n')

        for cand_set in result_candidates[1:]:
            results_file.write(str(sorted(cand_set))[1:-1] + '\n\n')

        results_file.write('Frequent Itemsets:\n')
        output = ''
        for single_item in sorted(result_frequent_itemsets[0]):
            output += '(\'' + str(single_item) + '\'),'

        results_file.write(output[:-1] + '\n\n')

        for freq_set in result_frequent_itemsets[1:]:
            results_file.write(str(sorted(freq_set))[1:-1] + '\n\n')


if __name__ == '__main__':
    start_time = time.time()
    conf = SparkConf()
    conf.set("spark.driver.memory", "4g")
    conf.set("spark.executor.memory", "4g")
    conf.setMaster('local[8]')
    conf.setAppName('Assignment_2')

    sc = SparkContext.getOrCreate(conf)
    argv = sys.argv

    data_path = 'asnlib/publicdata/'
    data = sc.textFile(data_path + argv[3]).map(lambda x: x.split(',')).map(lambda x: (x[0], x[1]))
    header = data.first()
    raw_data = data.filter(lambda x: x != header)

    if argv[1] == '2':
        raw_data = raw_data.map(lambda x: (x[1], x[0]))

    baskets = raw_data.distinct().groupByKey().map(lambda x: (x[0], sorted(list(x[1]))))
    print(baskets.collect())
    support = int(argv[2])
    partial_support = support // baskets.getNumPartitions()

    candidates = []
    frequent_itemsets = []
    set_size = 1
    while True:
        current_candidates = get_candidates_list(candidate_baskets=baskets)
        # current_candidates = []
        current_frequent_itemsets = get_original_frequent_sets(
            original_baskets=baskets,
            candidate_list=current_candidates
        )
        if not current_candidates:
            break

        candidates.append(current_candidates)
        if current_frequent_itemsets:
            frequent_itemsets.append(current_frequent_itemsets)

        set_size += 1

    write_results(result_candidates=candidates, result_frequent_itemsets=frequent_itemsets)
    # output 1
    print("Duration: {:.2f}".format(time.time() - start_time))
