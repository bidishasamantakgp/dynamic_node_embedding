
import picke

def load_data(filename):
    return pickle.load(filename)

def create_samples(args):
    input_data = load_data(args.data_file)
    train_data = input_data[:args.train_size]
    test_data = input_data[args.train_data:]
    samples_train = []
    samples_test = []

    for i in range(0, len(train_data), args.seq_length):
        sample = train_data[i : i + args.seq_length + 1]
        samples_train.append(sample)
    
    for i in range(0, len(test_data), args.seq_length):
        sample = test_data[i : i + args.seq_length + 1]
        samples_test.append(sample)

    return samples_train, samples_test

def next_sample(args, samples, i):
    reshaped = np.reshape(samples[i], len(samples[i]), 1)
    x = rehspaed[:,-1]
    y = reshaped[:1:]
return x, y