import findspark
findspark.init()

from pyspark import SparkContext, SparkConf
from itertools import combinations
import sys, time


def combine_list(a, b):
    if b not in a:
        return a + b


def ap(data, num_basket, support):

    threshold = support * (len(data) / num_basket)

    #print('partition threshold: ' + str(threshold))

    result = []

    count = {}

    l1 = []

    pair_count = {}

    # 1 pair

    for basket in data:

        basket.sort()

        for pair in combinations(basket, 2):

            if frozenset(pair) not in pair_count:

                pair_count[frozenset(pair)] = 1

            else:

                pair_count[frozenset(pair)] += 1

        for item in basket:

            if item not in count:

                count[item] = 1

                if count[item] >= threshold:

                    if (item,) not in l1:
                        l1.append(item)

            else:

                if count[item] < threshold:

                    count[item] += 1

                    if count[item] >= threshold:

                        if (item,) not in l1:
                            l1.append(item)

    # l1.sort()

    #print("num singleton: " + str(len(l1)))

    result.append((1, l1))

    # 2 pair

    l2 = []

    for pair in combinations(l1, 2):

        # print(pair)

        if frozenset(pair) in pair_count:

            if pair_count[frozenset(pair)] >= threshold:

                if set(pair) not in l2:

                    l2.append(set(pair))

    # l2.sort(key = lambda x : x[0])

    result.append((2, l2))

    candidate = l2

    comb = 3

    while candidate:

        # new combs

        new_candidate = []

        for i in range(len(candidate)):

            for j in range(i + 1, len(candidate)):

                new_comb = candidate[i].union(candidate[j])

                if len(new_comb) == comb:

                    if new_comb not in new_candidate:

                        new_candidate.append(new_comb)

        # c k

        ck = []

        for cand in new_candidate:

            cand_support = 0

            for basket in data:

                if cand.issubset(basket):

                    cand_support += 1

                    if cand_support >= threshold and cand not in ck:

                        ck.append(cand)

        # l k

        lk = []

        for cand in ck:

            is_frequent = True

            for subset in combinations(cand, comb - 1):

                if set(subset) not in result[comb - 2][1]:
                    # print(subset)
                    is_frequent = False

                    break

            if is_frequent and cand not in lk:
                #print(cand)
                lk.append(cand)

        #print('num of ' + str(comb) + ': ' + str(len(lk)))

        result.append((comb, lk))

        candidate = lk

        comb += 1

    tuple_result = []

    count = 1

    for eles in result:
        tuple_list = []
        for ele in eles[1]:
            if count == 1:
                ele = tuple((ele,))
            else:
                ele = tuple( sorted(tuple(ele)) )
            tuple_list.append(ele)
        tuple_result.append((count, tuple_list))
        count += 1
    return tuple_result


def file_output(file, cand):

    if len(cand[0]) == 1:

        cand.sort()

        single_pair_res = ''

        for ele in cand:
            single_pair_res += '(\'' + ele[0] + '\')' + ','

        file.write(single_pair_res[0:-1])

        file.write('\n')

    else:

        sorted_list = sorted(cand)

        other_pair_res = ''

        for ele in sorted_list:

            other_pair_res += str(tuple(ele)) + ','

        file.write(other_pair_res[0:-1])

        file.write('\n')

    file.write('\n')


if __name__ == '__main__':

    start = time.time()

    conf = SparkConf()
    conf.set("spark.driver.memory", "4g")
    conf.set("spark.executor.memory", "4g")
    conf.setMaster('local[8]')
    conf.setAppName('Assignment_2')

    sc = SparkContext.getOrCreate(conf)

    if len(sys.argv) < 4:
        print('Error: not enough statements')

        exit(-1)

    filter_threshold = int( sys.argv[1] )

    support = int( sys.argv[2] )

    task_path = sys.argv[3]

    output_path = sys.argv[4]

    test_data = sc.textFile(task_path)

    userBusinesses = test_data \
        .map(lambda x: x.split(',')) \
        .map(lambda x: (x[0], [x[1]])) \
        .reduceByKey(combine_list)

    firstLine = userBusinesses.first()

    case_data = userBusinesses \
        .filter(lambda x: x != firstLine) \
        .map(lambda x: (x[0], set(x[1]))) \
        .map(lambda x: (x[0], list(x[1]))) \
        .filter(lambda x: len(x[1]) > filter_threshold)

    # .persist(pyspark.StorageLevel(True, True, True, False, 1))

    case_data = sc \
        .parallelize(case_data.collect(), 2)

    basket_values = case_data.values()

    numBasket = basket_values.count()

    #print("num baskets : " + str(numBasket))

    # phase 1
    apRDD = basket_values \
        .mapPartitions(lambda x: ap(list(x), numBasket, support)) \
        .reduceByKey(lambda a, b: a + b) \
        .map(lambda x: set(x[1])) \
        .map(lambda x: list(x)) \
        .filter(lambda x: len(x) > 0) \
        .sortBy(lambda x: len(x[0]))

    candidate_itemsets = apRDD.collect()

    #print(len(candidate_itemsets))

    # phase 2

    def find_true_freq(data, phase_one_res):
        for candidates in phase_one_res:
            #print(candidates)
            for cand in candidates:
                s = set()
                for tu in cand:
                    s.add(tu)
                for basket in data:
                    if s.issubset(basket):
                        yield (cand, 1)


    sonRDD = basket_values \
        .mapPartitions(lambda x: find_true_freq(list(x), candidate_itemsets)) \
        .reduceByKey(lambda x, y: x + y) \
        .filter(lambda x: x[1] >= support) \
        .map(lambda x : (len(x[0]), [x[0]])) \
        .reduceByKey(lambda a, b : a + b) \
        .sortBy(lambda x: x[0])

    # sonRDD = basket_values \
    #     .mapPartitions(lambda x: find_true_freq(list(x), a)) \
    #     .reduceByKey(lambda x, y: x + y) \
    #     .filter(lambda x: x[1] >= support) \
    #     .sortBy(lambda x: len(x[0]))

    #
    frequent_itemsets = sonRDD.collect()

    output_file = open(output_path, 'w')

    output_file.write('Candidates:\n')

    for cand in candidate_itemsets:
        file_output(output_file, cand)

    output_file.write('Frequent Itemsets:\n')

    for fre in frequent_itemsets:
        cand = fre[1]

        file_output(output_file, cand)

    output_file.close()

    end = time.time()

    print('Duration: ' + str(end - start))
