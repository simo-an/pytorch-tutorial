from collections import OrderedDict

d1 = OrderedDict([('0', 'A'), ('1', 'B')])

for k, v in d1.items():
    print(f'{k}: {v}')

k1 = {'a': 1, 'b': 3}
k2 = {'b': 2, 'a': 4}

k_set = {}

k_set.update(k1)
k_set.update(k2)

print(k_set)

ra = (1.0, 2.0, 4.0) * 8

print(ra)