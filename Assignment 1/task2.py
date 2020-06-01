import json
import logging
import traceback
import findspark
findspark.init()

from pyspark import SparkContext


class TaskNoSpark:
    def __init__(self):
        try:
            path_data_dir = 'D:\\sachin\\MS\\USC\\Course Work\\INF 553 Data Mining\\Assignments\\Assignment 1\\asnlib\\publicdata\\'

            file = open(path_data_dir + 'review.json', 'r', encoding='utf-8')
            reviews_data = []
            for record in file:
                    reviews_data.append(json.loads(record))

            file = open(path_data_dir + 'business.json', 'r', encoding='utf-8')
            business_data = []
            for record in file:
                    business_data.append(json.loads(record))

            self.reviews_data = reviews_data
            self.business_data = business_data

        except Exception as e:
            raise e

    def get_categories(self):
        category_business_map = {}
        for business in self.business_data:
            if business.get('categories', None) is not None:
                all_categories = business.get('categories','').split(',')
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
                sum_rating += average_rating_map.get(business_id,{}).get('sum_rating', 0)
                rating_count += average_rating_map.get(business_id,{}).get('rating_count', 0)

            if rating_count != 0:
                category_business_map[category]['average_rating'] = sum_rating/rating_count

            else:
                category_business_map[category]['average_rating'] = 0

        return category_business_map

    def get_results(self, category_business_map):
        result = []
        for category in category_business_map:
            result.append([category, category_business_map[category]['average_rating']])

        return {"result": result.sort(key=lambda x: (-x[1],x[0]))}


class TaskWithSpark:
    def __init__(self):
        self.sc = SparkContext('local', 'Task_2')
        try:
            reviews_txt = self.sc.textFile('asnlib/publicdata/review.json')
            business_txt = self.sc.textFile('asnlib/publicdata/business.json')
            self.reviews_data = reviews_txt.map(lambda x: json.loads(x))
            self.business_data = business_txt.map(lambda x: json.loads(x))

        except Exception as e:
            raise e

    def get_business_ratings(self):
        return self.reviews_data.map(lambda x: (x['business_id'], x['stars']))

    def get_category_business(self):

if __name__ == '__main__':
    try:
        task_no_spark_obj = TaskNoSpark()
        category_business_map = task_no_spark_obj.get_categories()
        average_rating_map = task_no_spark_obj.get_average_rating_per_business()
        category_business_map = task_no_spark_obj.compute_average_rating_business_category(
            category_business_map=category_business_map,
            average_rating_map=average_rating_map
        )
        results = task_no_spark_obj.get_results(category_business_map)

    except Exception as e:
        logging.error(traceback.format_exc())


