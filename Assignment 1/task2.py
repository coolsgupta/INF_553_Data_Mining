import json
import logging
import traceback
import findspark
findspark.init()
import sys

from pyspark import SparkContext


class TaskNoSpark:
    def __init__(self):
        try:
            self.argv = sys.argv
            path_data_dir = '../resource/asnlib/publicdata/'

            file = open(path_data_dir + self.argv[1], 'r')
            reviews_data = []
            for record in file:
                reviews_data.append(json.loads(record))

            file = open(path_data_dir + self.argv[2], 'r')
            business_data = []
            for record in file:
                business_data.append(json.loads(record))

            self.reviews_data = reviews_data
            self.business_data = business_data

        except Exception as e:
            raise e

    def get_category_business_map(self):
        category_business_map = {}
        for business in self.business_data:
            if business.get('categories', None) is not None:
                all_categories = business.get('categories', '').split(',')
                for category in all_categories:
                    category = category.strip()
                    if category not in category_business_map:
                        category_business_map[category] = {
                            'associated_business': [],
                            'average_rating': 0
                        }
                    try:
                        category_business_map[category]['associated_business'].append(business['business_id'])

                    except:
                        logging.error('business id missing {}'.format(json.dumps(business)))
                        pass

        return category_business_map

    def get_average_rating_per_business(self):
        average_rating_map = {}
        for review in self.reviews_data:
            if review['business_id'] not in average_rating_map:
                average_rating_map[review['business_id']] = {
                    'sum_rating': 0,
                    'rating_count': 0,
                }

            average_rating_map[review['business_id']]['sum_rating'] += review['stars']
            average_rating_map[review['business_id']]['rating_count'] += 1

        return average_rating_map

    def compute_average_rating_business_category(self, category_business_map, average_rating_map):
        for category in category_business_map:
            sum_rating = 0
            rating_count = 0
            for business_id in category_business_map[category]['associated_business']:
                sum_rating += average_rating_map.get(business_id, {}).get('sum_rating', 0)
                rating_count += average_rating_map.get(business_id, {}).get('rating_count', 0)

            if rating_count != 0:
                category_business_map[category]['average_rating'] = sum_rating / rating_count

            else:
                category_business_map[category]['average_rating'] = 0

        return category_business_map

    def get_results(self, category_business_map):
        result = []
        for category in category_business_map:
            result.append([category, category_business_map[category]['average_rating']])

        result.sort(key=lambda x: (-x[1], x[0]))
        return {"result": result[:int(self.argv[5])]}

    def write_results(self, final_result):
        with open(self.argv[3], 'w') as write_file:
            write_file.write(json.dumps(final_result))
            write_file.close()


class TaskWithSpark:
    def __init__(self):
        try:
            self.sc = SparkContext('local[8]', 'task2')
            self.argv = sys.argv
            path_dir = '../resource/asnlib/publicdata/'
            reviews_txt = self.sc.textFile(path_dir + self.argv[1])
            business_txt = self.sc.textFile(path_dir + self.argv[2])
            self.reviews_data = reviews_txt.map(lambda x: json.loads(x))
            self.business_data = business_txt.map(lambda x: json.loads(x))

        except Exception as e:
            raise e

    @staticmethod
    def category_business_id_mapper(row):
        category_business_list = []
        for category in row['categories'].split(','):
            category_business_list.append((row['business_id'], category.strip()))
        return category_business_list

    def compute_average_rating_business_category(self):
        business_review_map = self.reviews_data.map(lambda x: (x['business_id'], x['stars']))
        categories_business_map = self.business_data.filter(lambda x: x.get('categories', None) is not None).flatMap(self.category_business_id_mapper)
        joined_rdd = categories_business_map.join(business_review_map)
        category_ratings_map = joined_rdd.map(lambda x: x[1]).groupByKey().map(lambda x: (x[0], list(x[1])))
        average_ratings = category_ratings_map.mapValues(lambda x: sum(x) / len(x)).sortBy(lambda x: (-x[1], x[0]))
        return average_ratings.take(int(self.argv[5]))

    def get_results(self, average_ratings):
        result = []
        for record in category_business_map:
            result.append([record[0], record[1]])

        return {"result": result.sort(key=lambda x: (-x[1], x[0]))}

    def write_results(self, final_result):
        with open(self.argv[3], 'w') as write_file:
            write_file.write(json.dumps(final_result))
            write_file.close()


if __name__ == '__main__':
    try:
        if sys.argv[4] == 'spark':
            task_spark = TaskWithSpark()
            category_average_ratings = task_spark.compute_average_rating_business_category()
            results = task_spark.get_results(category_average_ratings)
            task_spark.write_results(results)

        else:
            task_no_spark_obj = TaskNoSpark()
            category_business_map = task_no_spark_obj.get_category_business_map()
            average_rating_map = task_no_spark_obj.get_average_rating_per_business()
            category_business_map = task_no_spark_obj.compute_average_rating_business_category(
                category_business_map=category_business_map,
                average_rating_map=average_rating_map
            )
            task_no_spark_obj.write_results(task_no_spark_obj.get_results(category_business_map))

    except Exception as e:
        logging.error(traceback.format_exc())


