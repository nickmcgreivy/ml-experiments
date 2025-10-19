import pytest

from natural_language_processing.tokenizer.tokenizers import CharTokenizer, Vocab


def test_char_tokenizer():
    tokenizer = CharTokenizer()
    assert 'a' in tokenizer
    assert '1' in tokenizer
    assert '<unk>' in tokenizer
    assert all([len(token) == 1 for token in tokenizer.tokens if token != '<unk>'])
    assert tokenizer.encode(f'hello world 12345 {chr(256)}') == [char for char in 'hello world 12345 '] + ['<unk>']

def test_vocab():
    tokenizer = CharTokenizer()
    vocab = Vocab(tokenizer)
    assert 'a' in vocab
    assert '1' in vocab
    assert '<unk>' in vocab
    assert len(vocab) == len(tokenizer.tokens)
    assert vocab[0] == '<unk>'
    assert vocab[[0, 1, 2, 3]] == ['<unk>','\n', ' ', '0']
    assert vocab[vocab.token_to_idx('a')] == 'a'
    assert vocab[vocab.token_to_idx('<unk>')] == '<unk>'

@pytest.mark.parametrize('text, expected', [
    ('hello world', 'hello world'),
    (f'I am {chr(256)} feet tall', f'I am <unk> feet tall'),
    (f'some ascii characters are {chr(210)} {chr(211)} {chr(124)}',
     f'some ascii characters are {chr(210)} {chr(211)} {chr(124)}')
])
def test_vocab_encodings(text, expected):
    tokenizer = CharTokenizer()
    vocab = Vocab(tokenizer)
    assert vocab.decode(vocab.encode(text)) == expected
