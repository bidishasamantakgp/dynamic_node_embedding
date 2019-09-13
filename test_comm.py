import sys
from collections import defaultdict


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
            for v in range(n): 
                index = min(dict_recent_c[u], dict_recent_c[v])
                #index = event_dict[t_]
                t_ = samples[index][2]
                #print("Debug current", current, index)
                val = 0.0
                for i in range(index, current+1):
                    val += l_c_list[i][u][v][0]
                val *= (t - t_) * l_c_list[current]/ (current - index + 1)
                with open("comm.txt", "a") as fw:
			fw.write(str(t)+"\t"+str(v)+"\t"+str(val))
		#communication_res[t].append(val)
                if v == v1:
                    #most_recent_c.append(t_)
                    dict_recent_c[u] = current
                    dict_recent_c[v] = current
        
        #for association event
        
        else:
            for v in range(n):
                index = min(dict_recent_a[u], dict_recent_a[v])
                #index = event_dict[t_]
                t_ = samples[index][2]
                val = 0.0
                for i in range(index, current+1):
                    val += l_a_list[i][u][v][0]
                val *= (t - t_) * l_a_list[current]/ (current - index + 1)
                association_res[t].append(val)
                if v == v1:
                    #most_recent_a.append(t_)
                    dict_recent_a[u] = current
                    dict_recent_a[v] = current
        current += 1
    return communication_res, association_res

