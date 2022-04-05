from os.path import join, abspath, dirname
from Bio.Data import IUPACData

VERSION = '0.1.0'

# Directories
BASE_DIR = abspath(join(dirname(__file__), '..'))
CACHE_DIR = join(BASE_DIR, 'cache')
BIN_DIR = join(BASE_DIR, 'bin')
TEST_DATA_DIR = join(BASE_DIR, 'test', 'data')
MODELING_DIR = join(BASE_DIR, 'models')
MODELS_DIR = join(MODELING_DIR, 'compiled')
CODE_DIR = join(MODELING_DIR, 'stan_code')

# File paths
PLOTS_FORMAT = 'png'

NUCLEOTIDES = ['A', 'U', 'G', 'C']
COMPLEMENT = {'U': 'A', 'A': 'U', 'G': 'C', 'C': 'G', 'N': 'N',
              'T': 'A', '[': ']', ']': '['}

ALPHABET = ['A', 'B', 'C', 'D', 'E',
            'F', 'G', 'H', 'I', 'J',
            'K', 'L', 'M', 'N', 'O',
            'P', 'Q', 'R', 'S', 'T',
            'U', 'V', 'W', 'X', 'Y',
            'Z']

DNA_ALPHABET = ['A', 'C', 'G', 'T']
RNA_ALPHABET = ['A', 'C', 'G', 'U']
PROTEIN_ALPHABET = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                    'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
ALPHABET_N_ALLELES = {'dna': len(DNA_ALPHABET),
                      'rna': RNA_ALPHABET,
                      'protein': PROTEIN_ALPHABET}

PROT_AMBIGUOUS_VALUES = {'X': ''.join(PROTEIN_ALPHABET)}
PROT_AMBIGUOUS_VALUES.update(dict(zip(PROTEIN_ALPHABET, PROTEIN_ALPHABET)))
DNA_AMBIGUOUS_VALUES = IUPACData.ambiguous_dna_values
RNA_AMBIGUOUS_VALUES = IUPACData.ambiguous_rna_values

AMBIGUOUS_VALUES = {'dna': DNA_AMBIGUOUS_VALUES,
                    'rna': RNA_AMBIGUOUS_VALUES,
                    'protein': PROT_AMBIGUOUS_VALUES}

CMAP = 'viridis'
MAX_GENOTYPES = 2e7
U_MAX = 500
PHI_UB, PHI_LB = 100, 0
