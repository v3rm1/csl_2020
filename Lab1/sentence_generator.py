import numpy as np

lexicon = {
    'noun-hum': ['man', 'boy', 'woman', 'girl'],
    'noun-agress': ['dragon', 'monster', 'lion'],
    'noun-anim': ['cat', 'mouse', 'dog'] + ['man', 'boy', 'woman', 'girl'] + ['dragon', 'monster', 'lion'],
    'noun-food': ['sandwich', 'cookie', 'bread'],
    'noun-frag': ['glass', 'plate'],
    'noun-inanim': ['book', 'rock', 'car'] + ['sandwich', 'cookie', 'bread'] + ['glass', 'plate'],
    
    'verb-agpat': ['move', 'break'],
    'verb-percept': ['smell', 'see'],
    'verb-destroy': ['break', 'smash'],
    'verb-eat': ['eat'],
    'verb-intran': ['think', 'sleep', 'exist'],
    'verb-tran': ['see', 'chase', 'like'] + ['move', 'break', 'smell', 'smash'],
}

grammar = [
    ('noun-hum', 'verb-eat', 'noun-food'),
    ('noun-hum', 'verb-percept', 'noun-inanim'),
    ('noun-hum', 'verb-destroy', 'noun-frag'),
    ('noun-hum', 'verb-intran'),
    ('noun-hum', 'verb-tran', 'noun-hum'),
    ('noun-hum', 'verb-agpat', 'noun-inanim'),
    ('noun-hum', 'verb-agpat'),
    ('noun-anim', 'verb-eat', 'noun-food'),
    ('noun-anim', 'verb-tran', 'noun-anim'),
    ('noun-anim', 'verb-agpat', 'noun-inanim'),
    ('noun-anim', 'verb-agpat'),
    ('noun-inanim', 'verb-agpat'),
    ('noun-agress', 'verb-destroy', 'noun-frag'),
    ('noun-agress', 'verb-eat', 'noun-hum'),
    ('noun-agress', 'verb-eat', 'noun-anim'),
    ('noun-agress', 'verb-eat', 'noun-food'),
]

def sentence(template):
    return [np.random.choice(lexicon[t]) for t in template]

def generate_sentences(n):
    templates = np.random.choice(grammar, size=n)
    return list(map(sentence, templates))