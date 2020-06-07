import itertools
import findspark
findspark.init()
from pyspark import SparkContext, SparkConf
import sys

class SON:
    def __init__(self, argv):
        conf = SparkConf()
        conf.set("spark.driver.memory", "4g")
        conf.set("spark.executor.memory", "4g")
        conf.setMaster('local[8]')
        conf.setAppName('Assignment_2')

        self.sc = SparkContext.getOrCreate(conf)
        self.argv = argv

        data_path = 'asnlib/publicdata/'
        data = self.sc.textFile(data_path + self.argv[3])
        data = data.map(lambda x: x.split(',')).map(lambda x: (x[0], x[1]))
        header = data.first()
        self.raw_data = data.filter(lambda x: x != header)

        if self.argv[1] == '2':
            self.raw_data = self.raw_data.map(lambda x: (x[1],x[0]))

        self.baskets = self.raw_data.distinct().groupByKey()
        self.baskets = self.baskets.map(lambda x: (x[0], list(x[1])))
        print(self.baskets.collect())
        self.support = int(self.argv[2])
        self.partial_support = self.support//self.baskets.getNumPartitions()

    def get_partition_itemset_candidates(self, bucket, size_of_set):
        items = {}
        for basket in bucket:
            if len(basket[1]) < size_of_set:
                continue

            sets = itertools.combinations(basket[1], size_of_set)
            for item in sets:
                if item not in items:
                    items[item] = 0

                items[item] += 1

        for item in items:
            if items[item] >= self.partial_support:
                yield item

    def get_original_itemset_counts(self, basket, size_of_set):
        sets = itertools.combinations(basket[1], size_of_set)
        return list(map(lambda x: (x, 1), sets))

    def get_candidates_list(self, size_of_sets):
        candidate_sets = self.baskets.mapPartitions(lambda x: self.get_partition_itemset_candidates(x, size_of_sets))
        candidate_sets_list = candidate_sets.distinct().collect()
        return candidate_sets_list

    def get_original_frequent_pairs(self, candidate_list, size_of_sets):
        original_freq_sets = self.baskets\
            .flatMap(lambda x: self.get_original_itemset_counts(x, size_of_sets)) \
            .filter(lambda x: x[0] in candidate_list)\
            .reduceByKey(lambda a, b: a + b)\
            .filter(lambda x: x[1] >= self.support)\
            .collect()

        return original_freq_sets

    def write_results(self, result_candidates, result_frequent_itemsets):
        with open(self.argv[4]) as results_file:
            results_file.write('Candidates:\n')
            results_file.writelines(result_candidates)

            results_file.write('\nFrequent Itemsets:\n')
            results_file.writelines(result_frequent_itemsets)


if __name__ == '__main__':
    task_obj = SON(sys.argv)
    candidates = []
    frequent_itemsets = []
    set_size = 1
    while True:
        current_candidates = task_obj.get_candidates_list(size_of_sets=set_size)
        # current_candidates = []
        current_frequent_itemsets = task_obj.get_original_frequent_pairs(
            candidate_list=current_candidates,
            size_of_sets=set_size
        )
        if not current_frequent_itemsets:
            break

        candidates.append(current_candidates)
        frequent_itemsets.append(current_frequent_itemsets)
        set_size += 1

    task_obj.write_results(result_candidates=candidates, result_frequent_itemsets=frequent_itemsets)
