from collections import Counter

class Vocab:
    def __init__(self, tokens=[], min_freq=0, reserved_tokens=[]):
        # flatten 2D list of tokens if necessary
        if tokens and isinstance(tokens[0], list):
            tokens = [token for line in tokens for token in line]

        # create counter of token frequencies
        counter = Counter(tokens)
        self.token_freqs = sorted(counter.items(), reverse=True, key=lambda x: x[1])

        # idx_to_token: list based on sorted token frequencies
        # index of list represents index of token
        self.idx_to_token = sorted(set(['<unk>'] + reserved_tokens + [
            token for token, freq in self.token_freqs if freq >= min_freq]))

        # token_to_idx: invert the idx_to_token list
        self.token_to_idx = {token: idx 
                             for idx, token in enumerate(self.idx_to_token)}
    
    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        """Returns the index of the input tokens"""
        if isinstance(tokens, (list, tuple)):
            return [self.__getitem__(token) for token in tokens]
        else:
            return self.token_to_idx.get(tokens, self.unk)

    def to_tokens(self, indices):
        """Convert indices to list of tokens"""
        if hasattr(indices, '__len__') and len(indices) > 1:
            return [self.to_tokens(int(idx)) for idx in indices]
        else:
            return self.idx_to_token[int(indices)]
    
    @property
    def unk(self):
        """Gets the index of the unknown token"""
        return self.token_to_idx['<unk>']