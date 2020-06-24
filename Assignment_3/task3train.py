import findspark
import json
import itertools
import sys
import time
import random
import math

findspark.init()
from pyspark import SparkConf, SparkContext


def build_min_hash_func(a, b, p, m):
    def min_hash_func(x):
        return (((a * x + b) % p) % m)

    return min_hash_func


def get_min_hash_functions(num_func, buckets):
    list_a = random.sample(range(50331653, 92233720), num_func)
    list_b = random.sample(range(25165843, 92233720), num_func)
    p = 12582917
    min_hash_func_list = [build_min_hash_func(a, b, p, buckets) for a, b in zip(list_a, list_b)]

    return min_hash_func_list


def check_jaccard_similarity(candidate, business_user_tokens):
    business_set_1 = set(business_user_tokens.get(candidate[0], []))
    business_set_2 = set(business_user_tokens.get(candidate[1], []))
    pair_jac_sim = 0
    if business_set_1 and business_set_2:
        pair_jac_sim = len(business_set_1.intersection(business_set_2)) / len(business_set_1.union(business_set_2))
    return tuple([candidate, pair_jac_sim])


def write_results(results, file_path):
    with open(file_path, 'w') as file:
        for line in results:
            file.write(json.dumps(line) + '\n')
    file.close()


def computeSimilarity(dict1, dict2):
    """
    compute Pearson Correlation Similarity
    :param dict1:
    :param dict2:
    :return: a float number
    """
    co_rated_user = list(set(dict1.keys()) & (set(dict2.keys())))
    val1_list, val2_list = list(), list()
    [(val1_list.append(dict1[user_id]),
      val2_list.append(dict2[user_id])) for user_id in co_rated_user]

    avg1 = sum(val1_list) / len(val1_list)
    avg2 = sum(val2_list) / len(val2_list)

    numerator = sum(map(lambda pair: (pair[0] - avg1) * (pair[1] - avg2), zip(val1_list, val2_list)))

    if numerator == 0:
        return 0
    denominator = math.sqrt(sum(map(lambda val: (val - avg1) ** 2, val1_list))) * \
                  math.sqrt(sum(map(lambda val: (val - avg2) ** 2, val2_list)))
    if denominator == 0:
        return 0

    return numerator / denominator


if __name__ == '__main__':
    argv = sys.argv

    start_time = time.time()
    conf = SparkConf()
    conf.set("spark.driver.memory", "4g")
    conf.set("spark.executor.memory", "4g")
    conf.setMaster('local[8]')
    conf.setAppName('Assignment_3')
    sc = SparkContext.getOrCreate(conf)
    sc.setLogLevel("ERROR")

    # load data
    reviews_json = sc.textFile(argv[1]).map(json.loads)

    # create user business rating sets
    user_business_rating_sets = reviews_json\
        .map(lambda x: (x.get('user_id'), x.get('business_id'), x.get('stars'))).distinct()

    # create user tokens
    user_tokens = user_business_rating_sets \
        .map(lambda x: x[0]) \
        .distinct() \
        .sortBy(lambda x: x) \
        .zipWithIndex()

    user_tokens_dict = user_tokens.collectAsMap()

    inverse_user_tokens_dict = {bid: token for token, bid in user_tokens_dict.items()}

    # create business tokens
    candidate_business_tokens = user_business_rating_sets \
        .map(lambda x: x[1]) \
        .distinct() \
        .sortBy(lambda x: x) \
        .zipWithIndex()

    business_tokens_dict = candidate_business_tokens.collectAsMap()

    inverse_business_tokens_dict = {bid: token for token, bid in business_tokens_dict.items()}

    # create hash functions
    min_hash_func_list = get_min_hash_functions(50, len(user_tokens_dict) * 2)

    # get user business tokenized maps
    user_business_rating_tokenized_sets = user_business_rating_sets\
        .map(lambda x: (user_tokens_dict.get(x[0]), business_tokens_dict.get(x[1]), x[2]))

    # create user_business_tokenized_pairs
    business_user_tokenized_pairs = user_business_rating_tokenized_sets\
        .map(lambda x: (x[1], x[0]))

    # create business and rating user list
    business_user_tokenized_map = business_user_tokenized_pairs\
        .groupByKey()\
        .mapValues(lambda x: list(set(x)))\
        .filter(lambda x: len(x[1]) >= 3)

    # create user and rated business  list
    user_business_tokenized_dict = business_user_tokenized_map\
        .flatMap(lambda x: [(user, x[0]) for user in x[1]])\
        .groupByKey()\
        .mapValues(lambda x: list(set(x)))\
        .collectAsMap()

    if argv[3] == 'user_based':
        # create user business rating map
        user_business_rating_map_dict = user_business_rating_tokenized_sets\
            .map(lambda x: (x[0], (x[1], x[2])))\
            .groupByKey()\
            .mapValues(lambda x: {business_rating_pair[0]: business_rating_pair[1] for business_rating_pair in list(x)})\
            .collectAsMap()

        # create min_hash values for business tokens
        business_hashed_values = candidate_business_tokens\
            .map(lambda x: (x[1], [min_hash(x[1]) for min_hash in min_hash_func_list]))

        # create signature matrix for LSH
        signature_matrix_rdd = business_user_tokenized_map \
            .leftOuterJoin(business_hashed_values) \
            .map(lambda x: x[1]) \
            .flatMap(lambda user_set: [(x, user_set[1]) for x in user_set[0]]) \
            .reduceByKey(lambda a, b: [min(x, y) for x, y in zip(a, b)])

        # get candidate pairs by applying LSH
        candidate_pairs = signature_matrix_rdd \
            .flatMap(lambda x: [(tuple([i, tuple(x[1][i:i + 1])]), x[0]) for i in range(0, 30)]) \
            .groupByKey() \
            .map(lambda x: list(x[1])) \
            .filter(lambda val: len(val) > 1) \
            .flatMap(lambda uid_list: [pair for pair in itertools.combinations(uid_list, 2)])

        # filter pairs based on jaccard similarity >= 0.01
        jaccard_similar_users = candidate_pairs \
            .distinct() \
            .map(lambda x: check_jaccard_similarity(x, user_business_tokenized_dict)) \
            .filter(lambda x: x[1] >= 0.01)\

        # filter pairs based on positive pearson correlation
        pearson_similar_pairs = jaccard_similar_users\
            .map(lambda x: (x[0], computeSimilarity(user_business_rating_map_dict[x[0][0]], user_business_rating_map_dict[x[0][1]])))\
            .filter(lambda kv: kv[1] > 0)

        # final model in json format
        final_model = pearson_similar_pairs\
            .map(lambda kv: {"u1": inverse_user_tokens_dict[kv[0][0]], "u2": inverse_user_tokens_dict[kv[0][1]], "sim": kv[1]})\
            .collect()

    else:
        # create business user rating map
        business_user_rating_map = user_business_rating_tokenized_sets \
            .map(lambda x: (x[1], (x[0], x[2]))) \
            .groupByKey() \
            .mapValues(lambda x: {user_rating_pair[0]: user_rating_pair[1] for user_rating_pair in list(x)})

        business_user_rating_map_dict = business_user_rating_map.collectAsMap()

        # collect all business tokens
        candidate_business_tokens = business_user_rating_map.map(lambda x: x[0])

        # candidate pairs based on co-rated users
        filtered_candidate_pairs = candidate_business_tokens\
            .cartesian(candidate_business_tokens)\
            .filter(lambda x: x[0] < x[1])\
            .filter(lambda x: len(set(business_user_rating_map_dict.get(x[0], [])).intersection(set(business_user_rating_map_dict.get(x[1], [])))) >= 3)

        # filter pairs based on positive pearson correlation
        pearson_similar_pairs = filtered_candidate_pairs\
            .map(lambda x: (x, computeSimilarity(business_user_rating_map_dict[x[0]], business_user_rating_map_dict[x[1]])))\
            .filter(lambda x: x[1] > 0)

        # final model in json format
        final_model = pearson_similar_pairs \
            .map(lambda x: {"b1": inverse_user_tokens_dict[x[0][0]], "b2": inverse_user_tokens_dict[x[0][1]], "sim": x[1]}) \
            .collect()

    write_results(final_model, argv[2])


