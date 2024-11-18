# config.py

# Number of threads for encoding
NUM_THREADS = 2

# SIMD (NumPy) enabled/disabled
SIMD_ENABLED = True

# Choice
"""
1. Check an item (dictionary encoded)
2. Search by prefix (dictionary encoded)
3. Check an item (vanilla)
4. Search by prefix (vanilla)
"""
CHOICE = 2

# Item to look for (for 1 and 3)
item = "knvr"

# prefix to look for (for 2 and 4)
prefix = "knv"

