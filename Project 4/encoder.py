# encoder.py

from functools import partial

import config
import threading
import pickle
import time
import multiprocessing


def read_column_file(filename):
    with open(filename, 'r') as f:
        data = f.read().splitlines()
    return data


def build_dictionary(data):
    """
    Generate dictionary.
    """
    unique_items = list(set(data))
    dictionary = {item: idx for idx, item in enumerate(unique_items)}
    return dictionary


def save_encoded_file(dictionary, encoded_data, filename='encoded_column.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump({'dictionary': dictionary, 'encoded_data': encoded_data}, f)


def encode_chunk(chunk, dictionary):
    """
    Encode a chunk of data using the dictionary.
    """
    return [dictionary[item] for item in chunk]


def encode_data(data, dictionary, num_threads):
    """
    Encode data in parallel using multiple processes.
    """
    if not data:
        return []

    # Determine the size of each chunk
    chunk_size = len(data) // num_threads

    chunks = [data[i * chunk_size: (i + 1) * chunk_size] for i in range(num_threads)]

    remaining = len(data) % num_threads
    if remaining:
        chunks[-1].extend(data[-remaining:])

    # Create a partial function with the dictionary fixed
    encode_partial = partial(encode_chunk, dictionary=dictionary)

    # Use a multiprocessing Pool to encode chunks in parallel
    with multiprocessing.Pool(processes=num_threads) as pool:
        encoded_chunks = pool.map(encode_partial, chunks)

    # Combine the encoded chunks into a single list
    encoded_data = []
    for chunk in encoded_chunks:
        encoded_data.extend(chunk)

    return encoded_data


def main():
    data = read_column_file('column.txt')

    dictionary = build_dictionary(data)

    start_encode_time = time.time()

    encoded_data = encode_data(data, dictionary, config.NUM_THREADS)
    encode_time = time.time() - start_encode_time

    save_encoded_file(dictionary, encoded_data)

    print(f"Data encoded in {encode_time:.1f} seconds using {config.NUM_THREADS} threads")


if __name__ == '__main__':
    # main()
    data = read_column_file('column.txt')
    print(data[1000])

