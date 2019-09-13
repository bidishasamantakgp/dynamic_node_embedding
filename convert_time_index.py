import sys

import pickle

a = pickle.load(open(sys.argv[1]))
new_list = []


for (u, v, t, m) in a:
    new_list.append((u-1, v-1, t, m))

pickle.dump(new_list, open("event_total.pkl", "wb"))

