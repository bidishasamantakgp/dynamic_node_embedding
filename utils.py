def next_batch(args):
    t0 = np.random.randn(args.batch_size, 1, (2 * args.chunk_samples))
    mixed_noise = np.random.randn(
        args.batch_size, args.seq_length, (2 * args.chunk_samples)) * 0.1
    #x = t0 + mixed_noise + np.random.randn(
    #    args.batch_size, args.seq_length, (2 * args.chunk_samples)) * 0.1
    #y = t0 + mixed_noise + np.random.randn(
    #    args.batch_size, args.seq_length, (2 * args.chunk_samples)) * 0.1
    x = np.sin(2 * np.pi * (np.arange(args.seq_length)[np.newaxis, :, np.newaxis] / 10. + t0)) + np.random.randn(
        args.batch_size, args.seq_length, (2 * args.chunk_samples)) * 0.1 + mixed_noise*0.1
    y = np.sin(2 * np.pi * (np.arange(1, args.seq_length + 1)[np.newaxis, :, np.newaxis] / 10. + t0)) + np.random.randn(
        args.batch_size, args.seq_length, (2 * args.chunk_samples)) * 0.1 + mixed_noise*0.1

    y[:, :, args.chunk_samples:] = 0.
    x[:, :, args.chunk_samples:] = 0.
return x, y