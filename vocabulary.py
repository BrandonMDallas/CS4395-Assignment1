from collections import Counter

def build_vocab(train_file, min_freq=1):
    """
    Build vocabulary from the training file. Only words occurring at least
    `min_freq` times are kept.
    """
    vocab = Counter()
    with open(train_file, 'r', encoding='utf-8') as f:
        for line in f:
            vocab.update(line.strip().split())
    # Keep words that occur at least min_freq times
    vocab = {word for word, count in vocab.items() if count >= min_freq}
    return vocab

def replace_unknowns(file_path, vocab, output_file):
    """
    Replace words not in the given vocab with <UNK> and write the processed text to a new file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f, open(output_file, 'w', encoding='utf-8') as out_f:
            for line in f:
                words = line.strip().split()
                words = [word if word in vocab else '<UNK>' for word in words]
                out_f.write(' '.join(words) + '\n')
        print(f"Processed {file_path} and saved as {output_file}.")
    except FileNotFoundError:
        print(f"Warning: {file_path} not found. Skipping...")