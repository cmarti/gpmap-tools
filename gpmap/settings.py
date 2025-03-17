from os.path import join, abspath, dirname
from Bio.Data import IUPACData

# Directories
BASE_DIR = abspath(join(dirname(__file__), ".."))
BIN_DIR = join(BASE_DIR, 'gpmap', "bin")
DATASETS_DIR = join(BASE_DIR, 'gpmap', "datasets")
RAW_DATA_DIR = join(DATASETS_DIR, "raw")
PROCESSED_DIR = join(DATASETS_DIR, "data")
LANDSCAPES_DIR = join(DATASETS_DIR, "landscapes")
VIZ_DIR = join(DATASETS_DIR, "visualizations")

# File paths
PLOTS_FORMAT = "png"


ALPHABET = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
]
DNA_ALPHABET = ["A", "C", "G", "T"]
RNA_ALPHABET = ["A", "C", "G", "U"]
PROTEIN_ALPHABET = [
    "A",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "V",
    "W",
    "Y",
]

ALPHABETS = {
    "dna": DNA_ALPHABET,
    "rna": RNA_ALPHABET,
    "protein": PROTEIN_ALPHABET,
}

PROT_AMBIGUOUS_VALUES = {"X": "".join(PROTEIN_ALPHABET)}
PROT_AMBIGUOUS_VALUES.update(dict(zip(PROTEIN_ALPHABET, PROTEIN_ALPHABET)))
DNA_AMBIGUOUS_VALUES = IUPACData.ambiguous_dna_values
RNA_AMBIGUOUS_VALUES = IUPACData.ambiguous_rna_values
AMBIGUOUS_VALUES = {
    "dna": DNA_AMBIGUOUS_VALUES,
    "rna": RNA_AMBIGUOUS_VALUES,
    "protein": PROT_AMBIGUOUS_VALUES,
}

MAX_STATES = 2e7
U_MAX = 500

NUCLEOTIDES = ["A", "U", "G", "C"]
COMPLEMENT = {
    "U": "A",
    "A": "U",
    "G": "C",
    "C": "G",
    "N": "N",
    "T": "A",
    "[": "]",
    "]": "[",
}
