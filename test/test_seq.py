#!/usr/bin/env python
import unittest
import numpy as np
import pandas as pd

from gpmap.seq import (translate_seqs, guess_alphabet_type,
                           guess_space_configuration, get_custom_codon_table,
                           get_seqs_from_alleles, get_one_hot_from_alleles,
                           generate_freq_reduced_code, transcribe_seqs,
                           msa_to_counts, calc_allele_frequencies,
                           calc_expected_logp, calc_genetic_code_aa_freqs)


class SeqTests(unittest.TestCase):
    def test_guess_alphabet(self):
        alphabet_type = guess_alphabet_type(['A', 'C', 'G'])
        assert(alphabet_type == 'dna')
        
        alphabet_type = guess_alphabet_type(['A', 'C', 'G', 'T'])
        assert(alphabet_type == 'dna')
        
        alphabet_type = guess_alphabet_type([['A', 'C', 'G', 'T'],
                                             ['A', 'C', 'G', 'T']])
        assert(alphabet_type == 'dna')
        
        alphabet_type = guess_alphabet_type(['A', 'C', 'G', 'U'])
        assert(alphabet_type == 'rna')
        
        alphabet_type = guess_alphabet_type(['A', 'K', 'G', 'R'])
        assert(alphabet_type == 'protein')
        
        alphabet_type = guess_alphabet_type([['A', 'K'], ['G', 'R']])
        assert(alphabet_type == 'protein')
        
        alphabet_type = guess_alphabet_type(['0', '1'])
        assert(alphabet_type == 'custom')
        
    def test_guess_configuration(self):
        # Test with a simple scenario
        seqs = np.array(['AA', 'AB', 'AC',
                         'BA', 'BB', 'BC'])
        config = guess_space_configuration(seqs)
        assert(config['length'] == 2)
        assert(config['n_alleles'] == [2, 3])
        assert(config['alphabet_type'] == 'custom')
        
        # Fail when sequence space is not complete
        seqs = np.array(['AA', 'AB', 'AC',
                         'BA', 'BB'])
        try:
            config = guess_space_configuration(seqs)
            self.fail('guess_configuration did not fail with incomplete space')
        except ValueError:
            pass
        
        # Incomplete sequence space
        config = guess_space_configuration(seqs, ensure_full_space=False)
        assert(config['n_alleles'] == [2, 3])
        assert(config['alphabet'] == [['A', 'B'], ['A', 'B', 'C']])
        
        # Make complete landscape with all alleles
        config = guess_space_configuration(seqs, ensure_full_space=False, 
                                           force_regular=True)
        assert(config['n_alleles'] == [3, 3])
        assert(config['alphabet'] == [['0', 'A', 'B'], ['A', 'B', 'C']])
        
        # With different alleles per site but same number
        seqs = np.array(['AA', 'AC',
                         'BA', 'BC'])
        config = guess_space_configuration(seqs, ensure_full_space=False, 
                                           force_regular=True,
                                           force_regular_alleles=False)
        assert(config['n_alleles'] == [2, 2])
        assert(config['alphabet'] == [['A', 'B'], ['A', 'C']])
        
        # Make complete landscape with all alleles
        config = guess_space_configuration(seqs, ensure_full_space=False, 
                                           force_regular=True,
                                           force_regular_alleles=True)
        assert(config['n_alleles'] == [3, 3])
        assert(config['alphabet'] == [['A', 'B', 'C'], ['A', 'B', 'C']])
        
    def test_translate(self):
        dna = np.array(['ATGTGA', 'ATGTGAATG'])
        
        # Standard codon table
        protein = translate_seqs(dna, codon_table='Standard')
        assert(np.all(protein == ['M*', 'M*M']))
        
        # Test other codon tables with NCBI identifiers
        protein = translate_seqs(dna, codon_table=11)
        assert(np.all(protein == ['M*', 'M*M']))
    
    def test_get_custom_codon_table(self):
        aa = ['W', 'W', 'K', 'K', 'C', 'C', 'C', 'C',
              'H', 'H', '*', '*', 'I', 'I', '*', 'M',
              'K', 'K', 'K', 'K', 'E', 'E', 'E', 'E',
              'L', 'L', 'Q', 'Q', 'P', 'P', 'P', 'P',
              'D', 'D', 'D', 'S', 'A', 'A', 'A', 'A',
              'T', 'T', 'F', 'F', 'C', 'C', 'P', 'P',
              'V', 'V', 'V', 'V', 'G', 'G', 'G', 'G',
              'N', 'N', 'R', 'R', 'Y', 'Y', 'Y', 'Y']
        codons = ['UUU', 'UUC', 'UUA', 'UUG', 'UCU', 'UCC', 'UCA', 'UCG',
                  'UAU', 'UAC', 'UAA', 'UAG', 'UGU', 'UGC', 'UGA', 'UGG',
                  'CUU', 'CUC', 'CUA', 'CUG', 'CCU', 'CCC', 'CCA', 'CCG',
                  'CAU', 'CAC', 'CAA', 'CAG', 'CGU', 'CGC', 'CGA', 'CGG',
                  'AUU', 'AUC', 'AUA', 'AUG', 'ACU', 'ACC', 'ACA', 'ACG',
                  'AAU', 'AAC', 'AAA', 'AAG', 'AGU', 'AGC', 'AGA', 'AGG',
                  'GUU', 'GUC', 'GUA', 'GUG', 'GCU', 'GCC', 'GCA', 'GCG',
                  'GAU', 'GAC', 'GAA', 'GAG', 'GGU', 'GGC', 'GGA', 'GGG']
        aa_mapping = pd.DataFrame({'Letter': aa, 'Codon': codons})
        codon_table = get_custom_codon_table(aa_mapping)
        
        assert(codon_table.__str__())
        assert(codon_table.stop_codons == ['TAA', 'TAG', 'TGA'])

        # Ensure that translation works with the new code        
        dna = np.array(['ATGTGA', 'ATGTGATGG'])
        protein = translate_seqs(dna, codon_table=codon_table)
        assert(np.all(protein == ['S*', 'S*M']))
    
    def test_get_seqs_from_alleles(self):
        alleles = [['a', 'b'], ['0', '1']]
        seqs = list(get_seqs_from_alleles(alleles))
        assert(np.all(seqs == ['a0', 'a1', 'b0', 'b1']))
    
    def test_get_one_hot_from_alleles(self):
        alphabet = [['A', 'T'], ['C', 'G']]
        onehot = get_one_hot_from_alleles(alphabet)
        exp_onehot = np.array([[1, 0, 1, 0],
                               [1, 0, 0, 1],
                               [0, 1, 1, 0],
                               [0, 1, 0, 1]])
        assert(np.allclose(onehot.todense(), exp_onehot))
        
        alphabet = [['A', 'T', 'C'], ['C', 'G']]
        onehot = get_one_hot_from_alleles(alphabet)
        exp_onehot = np.array([[1, 0, 0, 1, 0],
                               [1, 0, 0, 0, 1],
                               [0, 1, 0, 1, 0],
                               [0, 1, 0, 0, 1],
                               [0, 0, 1, 1, 0],
                               [0, 0, 1, 0, 1]])
        assert(np.allclose(onehot.todense(), exp_onehot))
    
    def test_get_freq_reduced_dict(self):
        seqs = np.array(['AG', 'AA', 'CG', 'CA', 'TG', 'GC'])
        
        code = generate_freq_reduced_code(seqs, 2)
        assert(code[0]['A'] == 'A')
        assert(code[0]['C'] == 'C')
        assert(code[0]['G'] == 'X')
        assert(code[1]['A'] == 'A')
        assert(code[1]['G'] == 'G')
        assert(code[1]['C'] == 'X')
        
        # Changing allele names by new alphabet across all sites
        code = generate_freq_reduced_code(seqs, 2, keep_allele_names=False)
        assert(code[0]['A'] == 'A')
        assert(code[0]['C'] == 'B')
        assert(code[0]['G'] == 'X')
        assert(code[1]['A'] == 'B')
        assert(code[1]['G'] == 'A')
        assert(code[1]['C'] == 'X')
        
        # Using counts
        counts = np.array([1, 1, 1, 1, 5, 5])
        code = generate_freq_reduced_code(seqs, 2, counts=counts)
        assert(code[0]['A'] == 'X')
        assert(code[0]['C'] == 'X')
        assert(code[0]['G'] == 'G')
        assert(code[0]['T'] == 'T')
        assert(code[1]['A'] == 'X')
        assert(code[1]['G'] == 'G')
        assert(code[1]['C'] == 'C')
    
    def test_transcribe_seqs(self):
        seqs = np.array(['AG', 'AA', 'CG', 'CA', 'TG', 'GC'])
        code = generate_freq_reduced_code(seqs, 2)
        
        new_seqs = transcribe_seqs(seqs, code)
        assert(np.all(new_seqs == ['AG', 'AA', 'CG', 'CA', 'XG', 'XX']))
    
    def test_msa_to_counts(self):
        msa = np.array(['AGTGG',
                        'AGTGT',
                        'GGTGG',
                        'ATACC',
                        'ATACC',
                        'ATACG'])
        
        # Regular counting
        X, y = msa_to_counts(msa)
        assert(np.all(X == np.array(['AGTGG', 'AGTGT', 'ATACC', 'ATACG', 'GGTGG'])))
        assert(np.all(y == np.array([1, 1, 2, 1, 1])))
        
        # Count subsequences at some positions
        X, y = msa_to_counts(msa, positions=[1, 2])
        assert(np.all(X == np.array(['GT', 'TA'])))
        assert(np.all(y == np.array([3, 3])))
        
        # Count subsequences at some positions with phylogenetic correction
        X, y = msa_to_counts(msa, positions=[1, 2], phylo_correction=True, max_dist=0.15)
        assert(np.all(X == np.array(['GT', 'TA'])))
        assert(np.all(y == np.array([3., 2.])))
        
        X, y = msa_to_counts(msa, positions=[1, 2], phylo_correction=True, max_dist=0.25)
        assert(np.all(X == np.array(['GT', 'TA'])))
        assert(np.all(y == np.array([4/3., 1.])))
    
    def test_calc_allele_freqs(self):
        X = np.array(['AGCT', 'ACGT', 'GTCA'])
        allele_freqs = calc_allele_frequencies(X)
        for v in allele_freqs.values():
            assert(v == 0.25)
            
        X = np.array(['AGCT', 'ACGC', 'AAAA'])
        allele_freqs = calc_allele_frequencies(X)
        exp_freqs = {'A': 1/2, 'G': 1/6, 'C': 1/4, 'T': 1/12}
        for a, v in allele_freqs.items():
            assert(v == exp_freqs[a])
            
        # With sequence weights
        y = np.array([1, 1, 2])
        allele_freqs = calc_allele_frequencies(X, y=y)
        exp_freqs = {'A': 10/16, 'G': 2/16, 'C': 3/16, 'T': 1/16}
        for a, v in allele_freqs.items():
            assert(v == exp_freqs[a])
            
        # From genetic code
        allele_freqs = calc_genetic_code_aa_freqs()
        assert(allele_freqs['A'] == 4./61)
        assert(allele_freqs['L'] == 6./61)
        assert(allele_freqs['W'] == 1./61)
        
        allele_freqs = calc_genetic_code_aa_freqs(codon_table=11)
        assert(allele_freqs['A'] == 4./61)
        assert(allele_freqs['L'] == 6./61)
        assert(allele_freqs['W'] == 1./61)
    
    def test_calc_expected_logp(self):
        X = ['AGCT', 'ACGT', 'GTCA']
        exp_freqs = {'A': 1/4, 'G': 1/4, 'C': 1/4, 'T': 1/4}
        logq = calc_expected_logp(X, exp_freqs)
        assert(np.all(logq == -4 * np.log(4)))
    
        X = ['AGCT', 'ACGC', 'AAAA']    
        exp_freqs = {'A': 1/2, 'G': 1/6, 'C': 1/4, 'T': 1/12}
        logq = calc_expected_logp(X, exp_freqs)
        exp_logq = np.log([1 / (2 * 6 * 4 * 12),
                           1 / (2 * 4 * 4 * 6), 
                           1 / 2 ** 4])
        assert(np.allclose(logq, exp_logq))
        
        
if __name__ == '__main__':
    import sys;sys.argv = ['', 'SeqTests']
    unittest.main()
