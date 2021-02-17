import logging

# logging level
LOGGING_LEVEL = logging.DEBUG

# one hot encode the amino acids itself, map to position
POSITIONS_AMINO_ACIDS = {
    "A": 0,
    "C": 1,
    "D": 2,
    "E": 3,
    "F": 4,
    "G": 5,
    "H": 6,
    "I": 7,
    "K": 8,
    "L": 9,
    "M": 10,
    "N": 11,
    "P": 12,
    "Q": 13,
    "R": 14,
    "S": 15,
    "T": 16,
    "V": 17,
    "W": 18,
    "Y": 19,
}

# one hot encode features of amino acids, map to feature
FEATURES_AMINO_ACIDS = {
    # non-polar, aliphatic R groups
    "A": 0,
    "G": 0,
    "I": 0,
    "L": 0,
    "M": 0,
    "V": 0,

    # polar, uncharged R groups
    "C": 1,
    "N": 1,
    "P": 1,
    "Q": 1,
    "S": 1,
    "T": 1,

    # positively charged R groups
    "H": 2,
    "K": 2,
    "R": 2,

    # negatively charged R groups
    "D": 3,
    "E": 3,

    # non polar, aromatic R groups
    "F": 4,
    "W": 4,
    "Y": 4,
}

FEATURES_AMINO_ACIDS_LENGTH = len(set(FEATURES_AMINO_ACIDS.values()))
