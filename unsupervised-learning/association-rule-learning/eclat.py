# ECLAT Algorithm - ML
from collections import defaultdict
from itertools import combinations
transactions = {
    "T1": ["Bread", "Butter", "Jam"],
    "T2": ["Butter", "Coke"],
    "T3": ["Butter", "Milk"],
    "T4": ["Bread", "Butter", "Coke"],
    "T5": ["Bread", "Milk"],
    "T6": ["Butter", "Milk"],
    "T7": ["Bread", "Milk"],
    "T8": ["Bread", "Butter", "Milk", "Jam"],
    "T9": ["Bread", "Butter", "Milk"]
}
min_support = 2

def generate_tidsets(transactions):
    item_tidset = defaultdict(set)
    for tid, items in transactions.items():
        for item in items:
            item_tidset[item].add(tid)
    return item_tidset

item_tidset = generate_tidsets(transactions)

for item, tidset in item_tidset.items():
    print(item, ":", sorted(tidset))

items = sorted(item_tidset.items(), key=lambda x: len(x[1]))

def eclat(prefix, items, min_support, frequent_itemsets):
    """
    prefix: list of items forming the current prefix
    items: list of tuples (item, tidset) to consider for extension
    min_support: absolute minimum support (count)
    frequent_itemsets: dict to collect results {frozenset(itemset): support_count}
    """
    while items:
        item, tidset = items.pop()
        support = len(tidset)
        if support >= min_support:
            new_itemset = prefix + [item]
            frequent_itemsets[frozenset(new_itemset)] = support
            suffix = []
            for other_item, other_tidset in items:
                intersection = tidset & other_tidset
                if len(intersection) >= min_support:
                    suffix.append((other_item, intersection))
            suffix = sorted(suffix, key=lambda x: len(x[1]))
            eclat(new_itemset, suffix, min_support, frequent_itemsets)

item_tidset = generate_tidsets(transactions)
items = sorted(item_tidset.items(), key=lambda x: len(x[1]))
frequent_itemsets = {}
eclat([], items, min_support, frequent_itemsets)

print("Frequent itemsets (as list) -> support count")
for itemset, support in sorted(frequent_itemsets.items(), key=lambda x: (-len(x[0]), -x[1], sorted(list(x[0])))):
    print(list(itemset), "=>", support)

