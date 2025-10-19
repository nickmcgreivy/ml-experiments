from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Set, List, Union, Tuple

class Tokenizer(ABC):
    @abstractmethod
    def __init__(self):
        pass

    """Creates a set of tokens"""
    @abstractmethod
    def build_tokens(self):
        """Creates a set of tokens"""
        pass

    @property
    def tokens(self) -> Set[str]:
        if not hasattr(self, '_tokens'):
            raise RuntimeError("Must call build_tokens() before accessing tokens")
        return self._tokens
    
    def __contains__(self, token: str) -> bool:
        return token in self._tokens

    @abstractmethod
    def encode(self, text: str) -> List[str]:
        pass


class CharTokenizer(Tokenizer):
    UNK_TOKEN = '<unk>'
    EXCLUDED = {127, 129, 141, 143, 144, 157}
    ASCII_START = 32
    ASCII_END = 255
    ADDITIONAL_CHARS = [10]

    def __init__(self):
        self.build_tokens()

    """A character-level tokenizer using ASCII characters"""
    def build_tokens(self):
        ascii_nums = [i for i in range(self.ASCII_START, self.ASCII_END) 
                      if i not in self.EXCLUDED] + self.ADDITIONAL_CHARS
        ascii_chars = [chr(num) for num in ascii_nums]
        self._tokens = {self.UNK_TOKEN, *ascii_chars}
    
    def encode(self, text: str) -> List[str]:
        return [char if char in self._tokens else self.UNK_TOKEN for char in text]


class Vocab:
    """Creates a vocabulary over a given tokenizer"""
    def __init__(self, tokenizer: Tokenizer) -> None:
        self.UNK_TOKEN = tokenizer.UNK_TOKEN
        self.tokenizer = tokenizer
        self.build_vocab(tokenizer.tokens)

    def build_vocab(self, tokens: Set[str]) -> None:
        self._token_to_idx = {}
        self._idx_to_token = {}

        def sort_tokens(token: str) -> Tuple[int, str]:
            if token == self.UNK_TOKEN:
                return (0, '')
            elif token.isalnum() or token in {' ', '\n'}:
                return (1, token)
            else:
                return (2, token)

        for i, token in enumerate(sorted(tokens, key=sort_tokens)):
            self._token_to_idx[token] = i
            self._idx_to_token[i] = token
    
    def __len__(self) -> int:
        return len(self._token_to_idx)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={len(self)}, tokenizer={self.tokenizer.__class__.__name__})"
    
    def __contains__(self, token: str) -> bool:
        return token in self._token_to_idx

    def __getitem__(self, idx: Union[int, Iterable[int]]) -> Union[str, List[str]]:
        if isinstance(idx, int):
            if idx not in self._idx_to_token:
                raise KeyError(f"Index {idx} not in vocabulary")
            return self._idx_to_token[idx]
        elif isinstance(idx, Iterable):
            return [self.__getitem__(i) for i in idx]
        else:
            raise TypeError(f"Expected int or Iterable, got {type(idx)}")
            
    def token_to_idx(self, token: Union[str, List[str]]) -> Union[int, List[int]]:
        if isinstance(token, list):
            return [self.token_to_idx(t) for t in token]
        elif isinstance(token, str):
            return self._token_to_idx.get(token, self._token_to_idx[self.UNK_TOKEN])
        else:
            raise TypeError(f"Expected str or List[str], for {type(token)}")
        
    def idx_to_token(self, idx: Union[int, Iterable[int]]) -> Union[str, Iterable[str]]:
        return self[idx]
    
    def encode(self, text: str) -> List[int]:
        if not isinstance(text, str):
            raise TypeError(f"Expected str, got {type(text)}")
        return [self.token_to_idx(token) for token in self.tokenizer.encode(text)]

    def decode(self, indices: List[int]) -> str:
        return ''.join(self[indices])