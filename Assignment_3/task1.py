import findspark
import json
import itertools
import sys
import time
import random

findspark.init()
from pyspark import SparkConf, SparkContext


def build_min_hash_func(a, b, p, m):
    def min_hash_func(x):
        return (((a * x + b) % p) % m)

    return min_hash_func


def get_min_hash_functions(num_func, buckets):
    min_hash_func_list = []
    list_a = random.sample(range(1, sys.maxsize - 1), num_func)
    list_b = random.sample(range(0, sys.maxsize - 1), num_func)
    p = 233333333333

    for a, b in zip(list_a, list_b):
        min_hash_func_list.append(build_min_hash_func(a, b, p, buckets))

    return min_hash_func_list


#         min_hash_func_list.append(lambda x: ((a*x + b)%p)%buckets)
#         print (a,b)


def verifySimilarity(candidate_pairs_list, business_user_tokens, inverse_business_tokens, threshold):
    result = []
    for candidate in candidate_pairs_list:
        business_set_1 = set(business_user_tokens.get(candidate[0], []))
        business_set_2 = set(business_user_tokens.get(candidate[1], []))
        if business_set_1 and business_set_2:
            pair_jac_sim = len(business_set_1.intersection(business_set_2)) / len(business_set_1.union(business_set_2))
            if pair_jac_sim >= threshold:
                result.append(
                    {
                        "b1": inverse_business_tokens[candidate[0]],
                        "b2": inverse_business_tokens[candidate[1]],
                        "sim": pair_jac_sim
                     }
                )
    return result


if __name__ == '__main__':
    argv = sys.argv

    start_time = time.time()
    conf = SparkConf()
    conf.set("spark.driver.memory", "4g")
    conf.set("spark.executor.memory", "4g")
    conf.setMaster('local[8]')
    conf.setAppName('Assignment_3')
    sc = SparkContext.getOrCreate(conf)

    # load data
    reviews_json = sc.textFile('asnlib/publicdata/train_review.json').map(json.loads)

    # get all user_business_pairs
    user_business_pairs = reviews_json.map(lambda x: (x.get('user_id'), x.get('business_id'))).distinct()

    # create user tokens
    user_tokens = user_business_pairs\
        .map(lambda x: x[0])\
        .distinct()\
        .sortBy(lambda x: x)\
        .zipWithIndex()\

    user_tokens_dict = user_tokens.collectAsMap()

    # create business tokens
    business_tokens_dict = user_business_pairs\
        .map(lambda x: x[1])\
        .distinct()\
        .sortBy(lambda x: x)\
        .zipWithIndex()\
        .collectAsMap()

    # get user business tokenized maps
    user_business_tokenized_pairs = user_business_pairs\
        .map(lambda x: (user_tokens_dict.get(x[0]), business_tokens_dict.get(x[1])))\

    user_business_tokenized_map = user_business_tokenized_pairs\
        .groupByKey()\
        .map(lambda x: (x[0], list(set(x[1]))))

    business_user_tokenized_map = user_business_tokenized_pairs\
        .map(lambda x: (x[1], x[0]))\
        .groupByKey()\
        .map(lambda x: (x[0], list(set(x[1]))))

    business_user_tokenized_dict = business_user_tokenized_map.collectAsMap()

    # create inverse tokens
    inverse_business_tokens_dict = {bid: token for token, bid in business_tokens_dict.items()}

    # create hash functions
    min_hash_func_list = get_min_hash_functions(30, len(user_tokens_dict) * 2)

    # get hashed values for users
    user_hashed_values = user_tokens.map(lambda x: (user_tokens_dict.get(x[0]), [min_hash(x[1]) for min_hash in min_hash_func_list]))

    # create signature matrix
    signature_matrix_rdd = user_business_tokenized_map\
        .leftOuterJoin(user_hashed_values)\
        .map(lambda x: x[1])\
        .flatMap(lambda business_set: [(x, business_set[1]) for x in business_set[0]])\
        .reduceByKey(lambda a, b: [min(x, y) for x, y in zip(a, b)])

    # get candidate pairs
    candidate_pairs = signature_matrix_rdd \
        .flatMap(lambda x: [(tuple([i, hash(tuple(x[1][i:i + 1]))]), x[0]) for i in range(0, 30)]) \
        .groupByKey().map(lambda kv: list(kv[1])).filter(lambda val: len(val) > 1) \
        .flatMap(lambda bid_list: [pair for pair in itertools.combinations(bid_list, 2)])

    # find actual similar pairs
    result_list = verifySimilarity(candidate_pairs_list=set(candidate_pairs.collect()),
                                   business_user_tokens=business_user_tokenized_dict,
                                   inverse_business_tokens=inverse_business_tokens_dict,
                                   threshold=0.05)

    print('Duration: {:.2f}'.format(time.time() - start_time))

    print('completed')

