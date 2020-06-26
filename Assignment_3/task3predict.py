import findspark
import json
import sys
import time

findspark.init()
from pyspark import SparkConf, SparkContext


def write_results(results, file_path):
    with open(file_path, 'w') as file:
        for line in results:
            file.write(json.dumps(line) + '\n')
    file.close()


def item_based_model_predict(bid_user_ratings, n, item_model, business_average_ratings_dict, inverse_tokens_dict, over_all_avg):
    candidate_id = bid_user_ratings[0]
    rating_similarity_pairs = sorted(
        [
            tuple([
                business_rating_pair[1],
                item_model.get(tuple(sorted([candidate_id, business_rating_pair[0]])), 0)
            ])
            for business_rating_pair in list(bid_user_ratings[1])
        ],
        key=lambda x: x[1], reverse=True
    )

    n_similar_businesses = rating_similarity_pairs[:n]

    try:
        similarity_rating = sum(map(lambda x: x[0] * x[1], n_similar_businesses))
        if not similarity_rating:
            raise Exception('No similarity rating')

        return tuple([candidate_id, similarity_rating / sum(map(lambda x: abs(x[1]), n_similar_businesses))])

    except:
        return tuple(
            [candidate_id, business_average_ratings_dict.get(inverse_tokens_dict.get(candidate_id), over_all_avg)])


def user_based_model_predict(uid_business_ratings, user_model, user_average_ratings_dict, inverse_tokens_dict,
                             over_all_avg):
    candidate_id = uid_business_ratings[0]
    rating_similarity_sets = [
        tuple([
            user_rating_pair[1],
            user_average_ratings_dict.get(inverse_tokens_dict.get(user_rating_pair[0], ''), over_all_avg),
            user_model.get(tuple(sorted([candidate_id, user_rating_pair[0]])), 0)
        ])
        for user_rating_pair in list(uid_business_ratings[1])
    ]

    try:
        similarity_rating = sum(map(lambda x: (x[0] - x[1]) * x[2], rating_similarity_sets))
        if not similarity_rating:
            raise Exception('No similarity rating')

        return tuple(
            [
                candidate_id,
                user_average_ratings_dict.get(candidate_id, over_all_avg) + similarity_rating /
                sum(map(lambda item: abs(item[2]), rating_similarity_sets))
            ]
        )

    except:
        return tuple([candidate_id, user_average_ratings_dict.get(inverse_tokens_dict.get(candidate_id), over_all_avg)])


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
    train_reviews_json = sc.textFile(argv[1]).map(json.loads)
    test_reviews_json = sc.textFile(argv[2]).map(json.loads)

    # create user business rating sets
    train_user_business_rating_sets = train_reviews_json \
        .map(lambda x: (x.get('user_id'), x.get('business_id'), x.get('stars'))).distinct()

    # create user tokens
    user_tokens = train_user_business_rating_sets \
        .map(lambda x: x[0]) \
        .distinct() \
        .sortBy(lambda x: x) \
        .zipWithIndex()

    user_tokens_dict = user_tokens.collectAsMap()

    inverse_user_tokens_dict = {bid: token for token, bid in user_tokens_dict.items()}

    # create business tokens
    candidate_business_tokens = train_user_business_rating_sets \
        .map(lambda x: x[1]) \
        .distinct() \
        .sortBy(lambda x: x) \
        .zipWithIndex()

    business_tokens_dict = candidate_business_tokens.collectAsMap()

    inverse_business_tokens_dict = {bid: token for token, bid in business_tokens_dict.items()}

    # tokenize trains sets
    train_user_business_rating_sets_tokenized = train_user_business_rating_sets \
        .map(lambda x: (user_tokens_dict[x[0]], business_tokens_dict[x[1]], x[2]))

    # load test set
    test_user_business_pairs_tokenized = test_reviews_json \
        .map(lambda x: (user_tokens_dict.get(x['user_id'], None), business_tokens_dict.get(x['business_id'], None))) \
        .filter(lambda x: x[0] is not None and x[1] is not None)

    # load model from memory
    model = sc.textFile(argv[3]) \
        .map(json.loads)

    # get model keys
    keys = list(model.first().keys())

    # collect final model as similarity pairs
    model = model \
        .map(lambda x: (x[keys[0]], x[keys[1]], x[keys[2]]))

    if argv[5] == 'user_based':
        # tokenize the model
        model = model.map(lambda x: ((user_tokens_dict[x[0]], user_tokens_dict[x[1]]), x[2])).collectAsMap()
        # business and list of user and respective user ratings
        business_user_rating_sets = train_user_business_rating_sets_tokenized \
            .map(lambda x: (x[1], (x[0], x[2]))) \
            .groupByKey() \
            .map(lambda x: (x[0], [(user_rating[0], user_rating[1]) for user_rating in list(set(x[1]))]))

        # dictionary of average ratings for each user from the given file
        user_average_ratings = sc.textFile('asnlib/publicdata/user_avg.json') \
            .map(json.loads) \
            .map(lambda x: dict(x)) \
            .flatMap(lambda x: x.items()) \
            .collectAsMap()
        over_all_avg = sum(user_average_ratings.values()) / len(user_average_ratings)

        # user_average_ratings = user_average_ratings[0]

        # re-order the test set
        test_user_business_pairs_tokenized = test_user_business_pairs_tokenized \
            .map(lambda x: (x[1], x[0]))

        # making predictions
        results = test_user_business_pairs_tokenized \
            .leftOuterJoin(business_user_rating_sets) \
            .mapValues(
            lambda x: user_based_model_predict(x, model, user_average_ratings, inverse_user_tokens_dict, over_all_avg)) \
            .map(lambda x:
                 {
                     "user_id": inverse_user_tokens_dict[x[1][0]],
                     "business_id": inverse_business_tokens_dict[x[0]],
                     "stars": x[1][1]
                 }
                 ) \
            .collect()

    else:
        model = model.map(lambda x: ((business_tokens_dict[x[0]], business_tokens_dict[x[1]]), x[2])).collectAsMap()
        # users and list of rated businesses with respective ratings
        user_business_rating_sets = train_user_business_rating_sets_tokenized \
            .map(lambda x: (x[0], (x[1], x[2]))) \
            .groupByKey() \
            .map(lambda x: (x[0], [(business_rating[0], business_rating[1]) for business_rating in list(set(x[1]))]))

        # dictionary of average ratings for each business from the given file
        business_average_ratings = sc.textFile('asnlib/publicdata/business_avg.json') \
            .map(json.loads) \
            .map(lambda x: dict(x)) \
            .flatMap(lambda x: x.items()) \
            .collectAsMap()

        over_all_avg = sum(business_average_ratings.values()) / len(business_average_ratings)
        # business_average_ratings = business_average_ratings[0]

        results = test_user_business_pairs_tokenized \
            .leftOuterJoin(user_business_rating_sets) \
            .mapValues(
            lambda x: item_based_model_predict(x, 3, model, business_average_ratings, inverse_business_tokens_dict,
                                               over_all_avg)) \
            .map(lambda x:
                 {
                     "user_id": inverse_user_tokens_dict[x[0]],
                     "business_id": inverse_business_tokens_dict[x[1][0]],
                     "stars": x[1][1]
                 }
                 ) \
            .collect()

    write_results(results, argv[4])

    print('Duration: {:.2f}'.format(time.time() - start_time))

    print('completed')
