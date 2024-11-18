# query.py

import config
import pickle
import numpy as np
import time

from config import *


def load_encoded_file(filename='encoded_column.pkl'):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data['dictionary'], data['encoded_data']


def query_item(item, dictionary, encoded_data, encoded_array):
    start_time = time.time()
    if item in dictionary:
        code = dictionary[item]
        if config.SIMD_ENABLED:
            indices = np.where(encoded_array == code)[0]
        else:
            indices = [idx for idx, val in enumerate(encoded_data) if val == code]
        query_time = time.time() - start_time
        print(f"Item '{item}' found at indices: {indices}")
        print(f"Query time: {query_time:.2f} seconds")
    else:
        print(f"Item '{item}' not found.")


def query_prefix(prefix, dictionary, encoded_data, encoded_array):
    matching_items = [item for item in dictionary if item.startswith(prefix)]
    if matching_items:
        start_time = time.time()
        codes = [dictionary[item] for item in matching_items]
        if config.SIMD_ENABLED:
            indices_dict = {}
            for code, item in zip(codes, matching_items):
                indices = np.where(encoded_array == code)[0]
                indices_dict[item] = indices
        else:
            indices_dict = {}
            for code, item in zip(codes, matching_items):
                indices = [idx for idx, val in enumerate(encoded_data) if val == code]
                indices_dict[item] = indices
        query_time = time.time() - start_time
        with open("result.txt", "w") as file:
            for item, indices in indices_dict.items():
                file.write(f"Item '{item}' found at indices: {indices}\n")
        print(f"Query time: {query_time:.1f} seconds")
    else:
        print(f"No items found with prefix '{prefix}'.")


def vanilla_query_item(item, data):
    start_time = time.time()
    indices = [idx for idx, val in enumerate(data) if val == item]
    query_time = time.time() - start_time
    if indices:
        print(f"Item '{item}' found at indices: {indices}")
    else:
        print(f"Item '{item}' not found.")
    print(f"Vanilla query time: {query_time:.1f} seconds")


def vanilla_query_prefix(prefix, data):
    start_time = time.time()
    indices_dict = {}
    for idx, val in enumerate(data):
        if val.startswith(prefix):
            if val not in indices_dict:
                indices_dict[val] = []
            indices_dict[val].append(idx)
    query_time = time.time() - start_time
    if indices_dict:
        with open("result.txt", "w") as file:
            for item, indices in indices_dict.items():
                file.write(f"Item '{item}' found at indices: {indices}\n")
    else:
        print(f"No items found with prefix '{prefix}'.")
    print(f"Vanilla query time: {query_time:.1f} seconds")


def main():
    # Load encoded data
    dictionary, encoded_data = load_encoded_file()
    encoded_array = np.array(encoded_data)

    # For vanilla query, load original data
    with open('column.txt', 'r') as f:
        data = f.read().splitlines()

    if CHOICE == 1:
        query_item(item, dictionary, encoded_data, encoded_array)
    elif CHOICE == 2:
        query_prefix(prefix, dictionary, encoded_data, encoded_array)
    elif CHOICE == 3:
        vanilla_query_item(item, data)
    elif CHOICE == 4:
        vanilla_query_prefix(prefix, data)


if __name__ == '__main__':
    main()
