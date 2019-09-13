import pickle
import tensorflow as tf
import numpy as np
from sklearn.metrics import mean_absolute_error
from collections import defaultdict
import math

def find_error(true, predicted):
    return mean_absolute_error(true, predicted)

def get_expected_c(test_times, l_c_list):
    l = len(test_times)
    res = np.zeros(len(l_c_list[0]))
    res_array = []
    for i in range(l):
        res = np.add(res, test_times[l - i - 1] * np.array(l_c_list[l - i - 1]))
        res_array.append(res)
    res_array.reverse()
    return res_array


def get_expected_a(test_times, l_a_list, n):
    l = len(test_times)
    res = np.zeros([n, n])
    res_array = []
    
    for i in xrange(l-1, l-1000, -1):
        l_1 = np.reshape(l_a_list[i],[n,n])
        res = np.add(res, np.multiply(test_times[i], l_1))
        res_array.append(res)
    
    res_array.reverse()
    
    for i in range(0, l - 1000 + 1):
        l_1 = np.reshape(l_a_list[l - 1002 - i], [n, n])
        l_2 = np.reshape(l_a_list[l - i - 2], [n, n])

        res = np.subtract(np.add(l_1, res_array[i]), l_2)
        #res = np.subtract(np.add(l_a_list[l - 1002 - i], res_array[i]), l_a_list[l - i - 2])
        res_array.insert(0,res)
    #print("res_array:", len(res_array), res_array[0].shape )
    return res_array


def get_most_recent_time(samples, n, l_a_list, l_c_list):
    dict_recent_a = defaultdict(int)
    dict_recent_c = defaultdict(int)
    
    #most_recent_a = []
    #most_recent_c = []
    
    event_dict = defaultdict(int)
    communication_res = defaultdict(list)
    association_res = defaultdict(list)
    
    current = 0

    print("Size debug", len(l_c_list), l_c_list[0].shape)
    for (u,v1,t,m) in samples:
        # for communication event
        u = int(u)
        v1 = int(v1)
        event_dict[t] = current
        #print("Debug entry:", u, v1, t, m, current)
        #print("Debug event_dict:", event_dict)
        #print("Debug :", dict_recent_c)
        #print("Debug :", dict_recent_a)
        if m == 1:
	    com_list = []
            for v in range(n): 
                index = min(dict_recent_c[u], dict_recent_c[v])
                #index = event_dict[t_]
                t_ = samples[index][2]
                #print("Debug current", current, index)
                val = 0.0
                for i in range(index, current+1):
                    val += l_c_list[i][u][v][0]
                print("Debug l_c:", val, t_, t)
                
                val *= (t - t_ + current - index + 1) * l_c_list[current][u][v][0]/ (current - index + 1)
                #val *= (t - t_) * l_c_list[current][u][v][0]/ (current - index + 1)
                print("Val", val)
                com_list.append(str(math.log(val + 1e-100)))
		#with open("comm.txt", "a") as fw:
		#	fw.write(str(t)+"\t"+str(v)+"\t"+str(val))
  
                #communication_res[t].append(val)
                if v == v1:
                    #most_recent_c.append(t_)
                    #dict_recent_c[u] = current
                    dict_recent_c[v] = current
            dict_recent_c[u] = current
	    with open("comm.txt", "a") as fw:
		fw.write(' '.join(com_list)+"\n")


        #for association event
        else:
	    a_list = []
            for v in range(n):
                index = min(dict_recent_a[u], dict_recent_a[v])
                #index = event_dict[t_]
                t_ = samples[index][2]
                val = 0.0
                for i in range(index, current+1):
                    val += l_a_list[i][u][v][0]
                print("Debug l_a:", val, t_, t)
                val *= (t - t_ + current - index + 1) * l_a_list[current][u][v][0] * 1.0 / (current - index + 1)
                #association_res[t].append(val)
                #with open("association.txt", "a") as fw:
		#	fw.write(str(t)+"\t"+str(v)+"\t"+str(val))
                a_list.append(str(math.log(val + 1e-100)))
		if v == v1:
                    #most_recent_a.append(t_)
                    #dict_recent_a[u] = current
                    dict_recent_a[v] = current
	    dict_recent_a[u] = current
            with open("ass.txt", "a") as fw1:
			fw1.write(' '.join(a_list)+"\n")
            
	
        current += 1
    #return communication_res, association_res

def get_rank_new(samples, m):
    true_event = 0.0
    rank_total = 0.0
    count_hit = 0.0
    if m == 1:
        lines = open("comm.txt").readlines()
    else:
        lines = open("ass.txt").readlines()
    count = 0
    for (u,v,t,m1) in samples:
        if m1 != m:
            continue
        line = lines[count].strip()
        prob = [float(x) for x in line.split(" ")]
        indexes = np.flip(np.argsort(prob))
        rank = np.where(indexes == v) [0]
        if rank < 100:
            count_hit += 1
        rank_total += rank
        count += 1
    return (count_hit, rank_total, count)


def get_rank(samples, l_a_list, m):
    true_event = 0.0
    rank_total = 0.0
    count_hit = 0.0
    for (u,v,t,m1) in samples:
        if m1 != m:
            continue
        true_event+=1
        prob = l_a_list[t]
        indexes = np.flip(np.argsort(prob))
        rank = np.where(indexes == v) [0]
        if rank < 100:
            count_hit += 1
        rank_total += rank
    return (count_hit, rank_total, true_event)

def calculate_association(samples, l_a_list, m1, n):
    count_hit = 0.0
    true_event = 0
    rank_total  = 0.0
    #indicator = np.ones([n, n])
    occurance = np.zeros([n,n])
    for (s, l_a) in zip(samples, l_a_list):
        u = int(s[0])
        v = int(s[1])
        t = s[2]
        m = int(s[3])
        l_a = np.reshape(l_a, [n, n])
        if m == (1 - m1):
            continue
        true_event += 1
        indi_u = np.zeros([n])
        for x in range(n):
            if (occurance[u][x] == 0 or occurance[u][x] >= 70000 or x == v):
                indi_u[x] = 1
            if occurance[u][x] >= 70000:
                    occurance[u][x] = 1


        #indi_u[v] = 1
        #occurance[u][v] += 1
        print("Debug indi:", m1,u,v, occurance[u], indi_u, np.count_nonzero(indi_u), np.nonzero(indi_u))
        indexes = np.flip(np.argsort(np.multiply(l_a[u], indi_u)))
        #print("Debug indices", indexes, v)
        rank = np.where(indexes == v) [0]
        #if np.where(indexes == v)[0] <= 10 :
        occurance[u][v] += 1
        if rank < 10 :
            count_hit += 1
        rank_total += rank

        #indicator[u][v] = 0
    if true_event == 0:
        retval = 0
        rank_total = 0
    else:
        retval = count_hit 
        

    return (retval, rank_total, true_event)



def calculate_communication(samples, l_c_list, n):
    count_hit = 0.0
    true_event = 0
    rank_total = 0.0

    for (s, l_c) in zip(samples, l_c_list):
        u = s[0]
        v = s[1]
        t = s[2]
        m = s[3]
        l_c = np.reshape(l_c,[n,n])
        if m == 0:
            continue
        true_event += 1
        indexes = np.flip(np.argsort(l_c[u]))
        #print("Debug indices", indexes, v, np.where(indexes == v))
        rank = np.where(indexes == v) [0]
        if rank < 10 :
            count_hit += 1
        rank_total += rank
    if true_event == 0:
        retval = 0
        rank_total = 0.0
    else:
        retval = count_hit 
    return (retval, rank_total, true_event)

def dummy_kl_gaussgauss(args, mu_1, sigma_1, mu_2, sigma_2):
        k = np.zeros([args.n])
        k.fill(args.z_dim)

        sigma_2_sigma_1 = []
        sigma_mu_sigma = []
        det = []
        for i in range(args.n):
                    sigma_2_inv = np.linalg.inv(sigma_2[i])
                    sigma_2_sigma_1.append(np.trace(np.multiply( sigma_2_inv, sigma_1[i])))
                    mu_diff = np.subtract(mu_2[i], mu_1[i])
                    sigma_mu_sigma.append(np.matmul(np.matmul(np.transpose(mu_diff), sigma_2_inv), mu_diff))
                    det.append(np.log(np.maximum(np.linalg.det(sigma_2), 1e-09)) - np.log(np.maximum(np.linalg.det(sigma_1), 1e-09)))
        first_term = np.stack(sigma_2_sigma_1)
        second_term = np.stack(sigma_mu_sigma)
        third_term = np.stack(det)
        return -np.sum(0.5 *(first_term + second_term + (third_term) - k))

def dummy_features(adj, feature, k, n, d):
    w_in = np.zeros([k, d, d])
    w_in.fill(0.5)
    #tf.get_variable(name="w_in", shape=[k,d,d], initializer=tf.constant_initializer(0.5))
    #w_in = tf.Print(w_in,[w_in], message="my w_in-values:")
    output_list = []
    for i in range(k):
            if i > 0:
                output_list.append( np.multiply(np.transpose(np.matmul(w_in[i], np.transpose(feature))), np.matmul(adj, output_list[i-1])))
            else:
                output_list.append(np.transpose(np.matmul(w_in[i], np.transpose(feature))))
            print("Debug output_list")
            print(output_list[i])

    return np.reshape(output_list[-1],[n, 1, d])

def extract_time(args, samples):
    time = []
    #for sample in samples:
    for i in range(args.seq_length):
        time.append(samples[0][i][2])
    time = np.resize(np.stack(time), (args.batch_size, args.seq_length, 1))
    return time


def get_one_hot_features(n):
    return np.ones([n, 1])
    #return np.identity(n)

def get_shape(tensor):
    '''return the shape of tensor as list'''
    return tensor.get_shape().as_list()

def load_data(filename):
    return pickle.load(open(filename, 'rb'))

def starting_adj(args, start, samples):
    input_data = load_data(args.data_file)
    adj = np.zeros([args.n, args.n])

    for i in range(start):
    #for sample in samples:
        sample = input_data[i]
        u = int(sample[0] - 1)
        v = int(sample[1] - 1)
        m = sample[3]
        
        if m == 0:
            adj[u][v] += 1
            adj[v][u] += 1

    return adj

def create_samples(args):
    input_data = load_data(args.data_file)
    start = input_data[0][2]
    #start = 0
    input_data = [(int(u), int(v), ((t-start) * 1.0)/ (args.hours), m) for (u, v, t, m) in input_data]
    #input_data = [(int(u-1), int(v-1), ((t-start) * 1.0)/ (args.hours * 3600), m) for (u, v, t, m) in input_data]
    train_size = (args.seq_length + 1)* ((args.train_size - args.start)/ (args.seq_length + 1))
    train_data = input_data[args.start:train_size]
    test_data = input_data[train_size - 1 : train_size -1 + args.test_size]
    samples_train = []
    samples_test = []
    print(len(train_data))
    for i in range(0, len(train_data), args.seq_length):
        #if i + args.seq_length < len(train_data):

        sample = train_data[i : i + args.seq_length + 1]
        
        if len(sample) < args.seq_length + 1:
            l_e = sample[-1]
            sample.extend([l_e for el in range(args.seq_length + 1 - len(sample))])
        samples_train.append(sample)
        #i += args.seq_length

    for i in range(0, len(test_data), args.seq_length):
        sample = test_data[i : i + args.seq_length + 1]
        
        if len(sample) < args.seq_length + 1:
            l_e = sample[-1]
            sample.extend([l_e for el in range(args.seq_length + 1 - len(sample))])

        samples_test.append(sample)
        #i += args.seq_length

    return samples_train, samples_test

def get_adjacency_list(samples, adj, n):
    adj_list = []
    s = samples.shape[1]
    #s = len(samples)
    for i in range(s):
    #for sample in samples:
        sample = samples[0][i]
        u = int(sample[0])
        v = int(sample[1])
        m = sample[3]

        #print("debug u, v", u, v)
        if m == 0:
            adj[u][v] += 1
            adj[v][u] += 1
        adj_list.append(adj)
    return adj_list

def next_batch(args, samples, i):
    reshaped = np.reshape(samples[i], [args.batch_size, args.seq_length+1, -1])
    #print reshaped.shape, samples[i]
    #x = samples[i][:-1]
    x = reshaped[:, :-1, :]
    #print "x", x.shape
    #y = samples[i][1:]
    y = reshaped[:, 1:, :]
    #print "y", y.shape
    return x, y


def print_vars(string):
    '''print variables in collection named string'''
    print("Collection name %s"%string)
    print("    "+"\n    ".join(["{} : {}".format(v.name, get_shape(v)) for v in tf.get_collection(string)]))
