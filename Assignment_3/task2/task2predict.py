import findspark
import json
import math
import sys
import re
import string
import time

findspark.init()
from pyspark import SparkConf, SparkContext


def get_cos_sim(user_profile, business_profile):
    if user_profile and business_profile:
        set_user_profile = set(user_profile)
        set_business_profile = set(business_profile)
        return len(set_user_profile.intersection(set_business_profile))/ math.sqrt(len(set_user_profile)*len(set_business_profile))
    else:
        return 0


def write_results(results, file_path):
    with open(file_path, 'w') as file:
        for line in results:
            file.write(json.dumps(line) + '\n')
    file.close()


if __name__ == '__main__':
    argv = sys.argv

    start_time = time.time()
    conf = SparkConf()
    conf.set("spark.driver.memory", "4g")
    conf.set("spark.executor.memory", "4g")
    conf.setMaster('local[8]')
    conf.setAppName('Assignment_3')
    sc = SparkContext.getOrCreate(conf)

    # loading trained model
    system_model_params = sc.textFile(argv[2]).map(json.loads)

    # load user, business tokens and profiles
    user_tokens_dict = system_model_params\
        .filter(lambda x: x['key_type'] == 'user_token')\
        .map(lambda x: (x['id'], x['token']))\
        .collectAsMap()

    business_tokens_dict = system_model_params\
        .filter(lambda x: x['key_type'] == 'business_token')\
        .map(lambda x: (x['id'], x['token']))\
        .collectAsMap()

    business_profile_dict = system_model_params\
        .filter(lambda x: x['key_type'] == 'business_profile')\
        .map(lambda x: (x['id'], x['token']))\
        .collectAsMap()

    user_profile_dict = system_model_params\
        .filter(lambda x: x['key_type'] == 'user_profile')\
        .map(lambda x: (x['id'], x['token']))\
        .collectAsMap()

    # load test data
    # test_pairs = sc.textFile(argv[1])\
    #     .map(json.loads)\
    #     .map(lambda x: (user_tokens_dict.get(x['user_id'], None), business_tokens_dict.get(x['business_id'], None)))\
    #     .filter(lambda x: x[0] is not None and x[1] is not None)\
    #     .distinct()

    # getting inverse token dicts
    inverse_business_token_dict = {token: oid for oid, token in business_tokens_dict.items()}
    inverse_user_token_dict = {token: oid for oid, token in user_tokens_dict.items()}

    # compute cosine similarities
    sim_sets_filtered = sc.textFile(argv[1])\
        .map(json.loads)\
        .map(lambda x: (user_tokens_dict.get(x['user_id'], None), business_tokens_dict.get(x['business_id'], None)))\
        .filter(lambda x: x[0] is not None and x[1] is not None)\
        .distinct()\
        .map(lambda x: ((x[0], x[1]), get_cos_sim(user_profile_dict.get(x[0], None), business_profile_dict.get(x[1], None))))\
        .filter(lambda x: x[1] >= 0.01)

    sim_sets_dict = sim_sets_filtered.collectAsMap()

    # get all sets
    sim_sets_json = [
        {
            'user_id': inverse_user_token_dict[pair[0]],
            'business_id': inverse_business_token_dict[pair[1]],
            'sim': sim_sets_dict[pair]
         }
        for pair in sim_sets_dict
    ]

    # write final results to file
    write_results(sim_sets_json, argv[3])

    print('Duration: {:.2f}'.format(time.time() - start_time))

    print('completed')
