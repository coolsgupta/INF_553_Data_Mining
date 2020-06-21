import findspark
import json
import math
import sys
import re
import string
import time

findspark.init()
from pyspark import SparkConf, SparkContext


def create_word_count_dict(word_list):
    count_dict = {}
    for word in word_list:
        if word not in count_dict:
            count_dict[word] = 0
        count_dict[word] += 1
    return count_dict


def get_tf_scores(word_dict):
    max_freq = max(word_dict.values())
    tf_scores = {}
    for word in word_dict:
        tf_scores[word] = word_dict[word] / max_freq
    return tf_scores


def add_to_model(key_type, model_params):
    additional_model_params = [
        dict(key_type=key_type, id=id_token, token=model_params[id_token]) for id_token in model_params
    ]
    return additional_model_params


def write_model(model_params):
    with open('model.json', 'w') as file:
        for line in model_params:
            file.write(json.dumps(line) + '\n')
    file.close()


if __name__ == '__main__':
    start_time = time.time()
    conf = SparkConf()
    conf.set("spark.driver.memory", "4g")
    conf.set("spark.executor.memory", "4g")
    conf.setMaster('local[8]')
    conf.setAppName('Assignment_3')
    sc = SparkContext.getOrCreate(conf)

    # load reviews
    reviews_json = sc.textFile('asnlib/publicdata/train_review.json').map(json.loads)
    stop_words = set(word.strip() for word in open("asnlib/publicdata/stopwords"))

    # create user tokens
    users_token_dict = reviews_json \
        .map(lambda x: x.get('user_id')) \
        .distinct() \
        .sortBy(lambda x: x) \
        .zipWithIndex() \
        .collectAsMap()

    # create business tokens
    business_token_dict = reviews_json \
        .map(lambda x: x.get('business_id')) \
        .distinct() \
        .sortBy(lambda x: x) \
        .zipWithIndex() \
        .collectAsMap()

    # create business and review word pairs
    business_word_pairs = reviews_json \
        .map(lambda x: (x.get('business_id'), x.get('text'))) \
        .flatMap(lambda x: [(x[0], word) for word in re.split(r'[{}]'.format(string.whitespace + string.punctuation), x[1].lower())]) \
        .mapValues(lambda x: x.strip(r'[{}]'.format(string.punctuation + string.digits))) \
        .filter(lambda x: x[1] and x[1] != '' and x[1] not in string.ascii_lowercase and x[1] not in string.digits and x[1] not in stop_words)

    # create word tokens
    words_token_dict = business_word_pairs \
        .map(lambda x: x[1]) \
        .distinct() \
        .sortBy(lambda x: x) \
        .zipWithIndex() \
        .collectAsMap()

    # tokenize business word pairs
    business_words_tokenized_pairs = business_word_pairs \
        .map(lambda x: (business_token_dict[x[0]], words_token_dict[x[1]]))

    # grouping words associated with a business
    business_review_words = business_words_tokenized_pairs \
        .groupByKey() \
        .map(lambda x: (x[0], list(x[1])))

    # getting counts for words in business reviews
    business_review_word_counts = business_review_words \
        .map(lambda x: (x[0], create_word_count_dict(x[1])))

    # calculate tf scores
    business_tf_scores = business_review_word_counts \
        .map(lambda x: (x[0], get_tf_scores(x[1])))

    # getting occurrences of words in different documents/ businesses
    word_business_occurrences = business_words_tokenized_pairs \
        .map(lambda x: (x[1], x[0])) \
        .groupByKey() \
        .map(lambda x: (x[0], list(set(x[1]))))

    # calculate idf scores
    len_business_token_dict = len(business_token_dict)
    word_idf_scores = word_business_occurrences \
        .map(lambda x: (x[0], math.log(len_business_token_dict / len(x[1]), 2)))
    word_idf_score_dict = word_idf_scores.collectAsMap()

    # calculate tf idf scores
    business_tf_idf = business_tf_scores\
        .map(lambda x: (x[0], [(word, x[1][word] * word_idf_score_dict[word]) for word in x[1]]))
    business_tf_idf_dict = business_tf_idf.collectAsMap()

    #  create business profiles with top 200 words based on tf-idf values
    business_tokenized_profile = business_tf_idf\
        .mapValues(lambda x: [word_score_pair[0] for word_score_pair in sorted(x, key=lambda record: -record[1])[:200]])
    business_tokenized_profile_dict = business_tokenized_profile.collectAsMap()

    # create user profiles by joining associated business profiles
    user_profile = reviews_json \
        .map(lambda x: (users_token_dict[x.get('user_id')], business_token_dict[x.get('business_id')])) \
        .distinct() \
        .flatMapValues(lambda x: business_tokenized_profile_dict[x]) \
        .groupByKey().mapValues(lambda x: list(set(x)))
    user_tokenized_profile_dict = user_profile.collectAsMap()

    # create model
    model = []
    model.extend(add_to_model('user_token', users_token_dict))
    model.extend(add_to_model('business_token', business_token_dict))
    model.extend(add_to_model('word_token', words_token_dict))
    model.extend(add_to_model('business_profile', business_tokenized_profile_dict))
    model.extend(add_to_model('user_profile', user_tokenized_profile_dict))

    # write model to file
    write_model(model)

    # print time of execution
    print('Duration: {:.2f}'.format(time.time() - start_time))

    print('completed')



