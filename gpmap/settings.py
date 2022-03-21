from os.path import join, abspath, dirname

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

DNA_ALPHABET = ['A', 'C', 'G', 'T']
RNA_ALPHABET = ['A', 'C', 'G', 'U']
PROTEIN_ALPHABET = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                    'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
ALPHABET_N_ALLELES = {'dna': len(DNA_ALPHABET),
                      'rna': RNA_ALPHABET,
                      'protein': PROTEIN_ALPHABET}

CMAP = 'viridis'
MAX_GENOTYPES = 2e7
U_MAX = 500
PHI_UB, PHI_LB = 100, 0
