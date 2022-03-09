recursion_depth = 0
levels = []

def inspect_dict(d):
    global recursion_depth
    for key, value in d.items():
        print(f"\t" * recursion_depth, key)
        if type(value) == dict:
            recursion_depth += 1
            levels.append((recursion_depth, key, len(value.keys())))
            inspect_dict(value)
        else:
            print(f"\t" * recursion_depth + "\t", value)


test_dict = {'a': {'b': [1, 2, 3], 'c': {'d': [4, 5, 6]}}}

inspect_dict(test_dict)
print(levels)
