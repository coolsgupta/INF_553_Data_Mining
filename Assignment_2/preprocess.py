import json
import csv
import findspark
findspark.init()
from pyspark import SparkContext, SparkConf

if __name__ == '__main__':
    conf = SparkConf()
    conf.set("spark.driver.memory", "4g")
    conf.set("spark.executor.memory", "4g")
    conf.setMaster('local[8]')
    conf.setAppName('Assignment_2')
    sc = SparkContext.getOrCreate(conf)

    path_dir = 'asnlib/publicdata/'
    reviews_json = sc.textFile(path_dir + 'review.json').map(json.loads)
    business_json = sc.textFile(path_dir + 'business.json').map(json.loads)

    required_business_ids = dict.fromkeys(
        business_json.filter(lambda x: x.get('state','') == 'NV').map(lambda x: x.get('business_id', '')).collect(),
        1
    )
    user_business_map = reviews_json.map(lambda x: (x.get('user_id', ''), x.get('business_id', ''))).filter(lambda x: x[1] in required_business_ids)

    csv_records = [('user_id', 'business_id')]
    csv_records.extend(user_business_map.collect())

    with open('data_file_2.csv', 'w', newline='') as data_file:
        writer = csv.writer(data_file)
        writer.writerows(csv_records)
        data_file.close()

