#!/usr/bin/env python
#!encoding=utf-8
import os
import re
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from scipy import stats
from joblib.parallel import Parallel, delayed
from sklearn.feature_extraction.text import CountVectorizer
dga_suffix_file = '/data1/new_workspace/mlxtend_dga_multi_20190316/merge/demo/public_suffix.txt'
# /home/soft/resource/aimatrix/dga/data/public_suffix.txt


class UnsuitableFeatureOrderException(Exception):
    pass


class PublicSuffixes:
    """
    Represents the official public suffixes list maintained by Mozilla  https://publicsuffix.org/list/
    """
    def __init__(self):
        with open(dga_suffix_file, 'r') as f:
            self.data = f.readlines()

        self.data = self.clean_domain_list(self.data)
        self.data = ['.' + s for s in self.data if not (s.startswith('/') or s.startswith('*'))]
        self.data = self.clean_domain_list(self.data)

    def get_valid_tlds(self):
        return [s for s in self.data if len(s.split('.')) == 2]

    def get_valid_public_suffixes(self):
        return self.data

    def clean_domain_list(self, domain_list, dga=False):
        """
        Cleans a given domain list from invalid domains and cleans each single domain in the list.
        :param domain_list:
        :param dga:
        :return:
        """
        domain_list = [d.strip().lower() for d in domain_list]
        domain_list = list(filter(None, domain_list))
    
        if dga:
            # some ramnit domains ending with the pattern: [u'.bid', u'.eu']
            to_remove = []
            for d in domain_list:
                if '[' in d:
                    to_remove.append(d)
                    res = set()
                    bracket_split = d.split('[')
                    tlds = bracket_split[1].split(',')
                    for tld in tlds:
                        tld = tld.strip()
                        tld = tld.split("'")[1].replace('.', '')
                        res_d = bracket_split[0] + tld
                        res.add(res_d)
                        domain_list.append(res_d)
    
            domain_list = [d for d in domain_list if d not in to_remove]
    
        return domain_list

class DGAExtractFeatures(object):
    def __init__(self):
        PUBLIC_SUF = PublicSuffixes()
        self.VALID_TLDS = PUBLIC_SUF.get_valid_tlds()  # TLD source https://publicsuffix.org/
        self.VALID_PUB_SUFFIXES = PUBLIC_SUF.get_valid_public_suffixes()
        self.__domain = ''
        self.__dot_split = ()
        self.__joined_dot_split = ''
        self.__dot_split_suffix_free = ()
        self.__joined_dot_split_suffix_free = ''
        self.__public_suffix = ''
        self.__unigram = ()
        self.ALL_FEATURES = self._length,  self._contains_digits, self._subdomain_lengths_mean, self._n_grams, self._hex_part_ratio, \
                  self._alphabet_size, self._shannon_entropy, self._consecutive_consonant_ratio

    def extract_features(self, d, features):
        """
        Extract all features given as arguments from the given domain
        :param features: arbitrary many features, names of the public functions
        :param d: domain name as string
        :param debug: set to true, to return a tuple of a scaled and unscaled feature vector
        :return: scaled feature vector according to input data (in case debug is set to True: feature_vector, scaled_vector)
        """
        feature_vector = []
    
        self.__fill_cache(d)
    
        # using exception here for more robustness due to defect data + performance better than if else statements
        for f in features:
            try:
                feature_vector = feature_vector + f()
            except (ValueError, ArithmeticError) as e:
                print('Feature {!s} could not be extracted of {!s}. Setting feature to zero'.format(f, d))
                feature_vector = feature_vector + [0]
        return feature_vector
    
    
    def __fill_cache(self, domain):
        self.__domain = domain
        self.__dot_split = tuple(domain.split('.'))
        self.__joined_dot_split = ''.join(list(self.__dot_split))
        self.__dot_split_suffix_free, self.__public_suffix = self.__public_suffix_remover(self.__dot_split)
        self.__joined_dot_split_suffix_free = ''.join(self.__dot_split_suffix_free)
    
    
    def __public_suffix_remover(self, dot_split):
        """
        Finds the largest matching public suffix
        :param dot_split: 
        :return: public suffix free domain as dot split, public suffix
        """
        match = ''
        if len(dot_split) < 2:
            return tuple(dot_split), match
    
        for i in range(0, len(dot_split)):
            sliced_domain_parts = dot_split[i:]
            match = '.' + '.'.join(sliced_domain_parts)
            if match in self.VALID_PUB_SUFFIXES:
                cleared = dot_split[0:i]
                return tuple(cleared), match
        return tuple(dot_split), match
    
    
    def _vowel_ratio(self):
        """
        Ratio of vowels to non-vowels
        :return: vowel ratio
        """
        vowel_count = 0
        alpha_count = 0
        domain = self.__joined_dot_split_suffix_free
        VOWELS = set('aeiou')
        for c in domain:
            if c in VOWELS:
                vowel_count += 1
            if c.isalpha():
                alpha_count += 1
    
        if alpha_count > 0:
            return [vowel_count/alpha_count]
        else:
            return [0]
    
    
    def _digit_ratio(self):
        """
        Determine ratio of digits to domain length
        :return:
        """
        domain = self.__joined_dot_split_suffix_free
        digit_count = 0
        for c in domain:
            if c.isdigit():
                digit_count += 1
        return [digit_count/len(domain)]
    
    
    def _length(self):
        """
        Determine domain length
        :return:
        """
        return [len(self.__domain)]
    
    
    def _contains_wwwdot(self):
        """
        1 if 'www. is contained' 0 else
        :return:
        """
        if 'www.' in self.__domain:
            return [1]
        else:
            return [0]
    
    
    def _contains_subdomain_of_only_digits(self):
        """
        Checks if subdomains of only digits are contained.
        :return: 
        """
        for p in self.self.__dot_split:
            only_digits = True
            for c in p:
                if c.isalpha():
                    only_digits = False
                    break
            if only_digits:
                return [1]
        return [0]
    
    
    def _subdomain_lengths_mean(self):
        """
        Calculates average subdomain length
        :return:
        """
        overall_len = 0
        for p in self.__dot_split_suffix_free:
            overall_len += len(p)
        if len(self.__dot_split_suffix_free) == 0:
            return [0]
        else:
            return [overall_len / len(self.__dot_split_suffix_free)]
    
    
    def _parts(self):
        """
        Calculate the number of domain levels present in a domain, where rwth-aachen.de evaluates to 1 -> [1,0,0,0,0]
        The feature is decoded in a binary categorical way in the form [0,0,0,1,0]. The index represents the number of subdomains
        :return:
        """
    
        PARTS_MAX_CONSIDERED = 4
        feature = [0] * PARTS_MAX_CONSIDERED
        split_length = len(self.__dot_split_suffix_free)
        if split_length >= PARTS_MAX_CONSIDERED:
            feature[PARTS_MAX_CONSIDERED - 1] = 1
        else:
            feature[split_length - 1] = 1
    
        return feature
    
    
    def _contains_ipv4_addr(self):
        """
        check if the domain contains a valid IP address. Considers both, v4 and v6
        :return:
        """
        # Pattern matching ipv4 addresses according to RFC
        ipv4_pattern = "(([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])\.){3}([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])"
        match_v4 = re.search(ipv4_pattern, self.__domain)
        if match_v4:
            return [1]
        else:
            return [0]
    
    
    def _contains_digits(self):
        if any(char.isdigit() for char in self.__domain):
            return [1]
        else:
            return [0]
    
    
    def _has_valid_tld(self):
        """
        Checks if the domain ends with a valid TLD
        :return:
        """
        if self.__public_suffix:
            return [1]
        return [0]
    
    
    def _contains_one_char_subdomains(self):
        """
        Checks if the domain contains subdomains of only one character
        :return:
        """
        parts = self.__dot_split
    
        if len(parts) > 2:
            parts = parts[:-1]
    
        for p in parts:
            if len(p) == 1:
                return [1]
    
        return [0]
    
    
    def _prefix_repetition(self):
        """
        Checks if the string is prefix repeating exclusively.
        Example: 123123 and abcabcabc are prefix repeating 1231234 and ababde are not.
        :return: 
        """
        i = (self.__domain + self.__domain).find(self.__domain, 1, -1)
        return [0] if i == -1 else [1]
    
    
    def _char_diversity(self):
        """
        counts different characters, divided by domain length. 
        :return: 
        """
        counter = defaultdict(int)
    
        domain = self.__joined_dot_split_suffix_free
        for c in domain:
            counter[c] += 1
        return [len(counter)/len(domain)]
    
    
    def _contains_tld_as_infix(self):
        """
        Checks for infixes that are valid TLD endings like .de in 123.de.rwth-aachen.de
        If such a infix is found 1 is returned, 0 else
        :return:
        """
        for tld in self.VALID_TLDS:
            if tld[1:] in self.__dot_split_suffix_free:
                return [1]
        return [0]
    
    
    def _n_grams(self):
        """
        Calculates various statistical features over the 1-,2- and 3-grams of the suffix and dot free domain
        :return: 
        """
    
        ngram_vectorizer = CountVectorizer(analyzer='char', ngram_range=(1, 1))
        counts = ngram_vectorizer.build_analyzer()(self.__joined_dot_split_suffix_free)
        npa = np.array(list(Counter(counts).values()), dtype=int)
        self.__unigram = npa
        feature = self.__stats_over_n_grams(npa)
        return feature
    
    def __stats_over_n_grams(self,npa):
        """
        Calculates statistical features over ngrams decoded in np arrays
        stddev, median, mean, min, max, quartils, alphabetsize (length of the ngram)
        :param npa: 
        :return: 
        """
        if npa.size > 0:
            stats = [npa.std(), np.median(npa), np.max(npa)]
        else:
            stats = [-1, -1, -1]
    
        return stats
    
    
    def _alphabet_size(self):
        """
        Calculates the alphabet size of the domain
        :return: 
        """
        if self.__unigram is ():
            raise UnsuitableFeatureOrderException('The feature _n_grams has to be calculated before.')
        return [len(self.__unigram)]
    
    
    def _shannon_entropy(self):
        """
        Calculates the Shannon Entropy based on the frequencies of characters.
        :return: 
        """
        # Note for me: checked against an online calculator for verification: http://www.shannonentropy.netmark.pl/
        if self.__unigram is ():
            raise UnsuitableFeatureOrderException('The feature _n_grams has to be calculated before.')
    
        return [stats.entropy(self.__unigram, base=2)]
    
    
    def _hex_part_ratio(self):
        """
        Counts all parts that are only hex. Normalized by the overall part count
        :return: 
        """
        hex_parts = 0
        HEX_DIGITS = set('0123456789abcdef')
        for p in self.__dot_split_suffix_free:
            if all(c in HEX_DIGITS for c in p):
                hex_parts += 1
        if len(self.__dot_split_suffix_free) == 0:
            return [0]
        else:
            return[hex_parts / len(self.__dot_split_suffix_free)]
    
    
    def _underscore_ratio(self):
        """
        Calculates the ratio of occuring underscores in all domain parts excluding the public suffix
        :return: 
        """
        underscore_counter = 0
        for c in self.__joined_dot_split_suffix_free:
            if c == '_':
                underscore_counter += 1
        if len(self.__joined_dot_split_suffix_free) == 0:
            return [0]
        else:
            return [underscore_counter / len(self.__joined_dot_split_suffix_free)]
    
    
    def _ratio_of_repeated_chars(self):
        """
        Calculates the ratio of characters repeating in the string
        :return: 
        """
        # TODO maybe weighted? check the impact
        if self.__unigram is ():
            raise UnsuitableFeatureOrderException('The feature _n_grams has to be calculated before.')
    
        repeating = 0
        for i in self.__unigram:
            if i > 1:
                repeating += 1
        if len(self.__unigram) == 0:
            return [0]
        else:
            return [repeating / len(self.__unigram)]
    
    
    def _consecutive_consonant_ratio(self):
        """
        Calculates the ratio of conescutive consonants
        :return: 
        """
        # TODO weighted: long sequences -> higher weight
    
        consecutive_counter = 0
        VOWELS = set('aeiou')
        for p in self.__dot_split_suffix_free:
            counter = 0
            i = 0
            for c in p:
                if c.isalpha() and c not in VOWELS:
                    counter +=1
                else:
                    if counter > 1:
                        consecutive_counter += counter
                    counter = 0
                i += 1
                if i == len(p) and counter > 1:
                    consecutive_counter += counter
        if len(self.__joined_dot_split_suffix_free) == 0:
            return [0]
        else:
            return [consecutive_counter / len(self.__joined_dot_split_suffix_free)]
    
    
    def _consecutive_digits_ratio(self):
        """
        Calculates the ratio of consecutive digits
        :return: 
        """
    
        consecutive_counter = 0
        for p in self.__dot_split_suffix_free:
            counter = 0
            i = 0
            for c in p:
                if c.isdigit():
                    counter +=1
                else:
                    if counter > 1:
                        consecutive_counter += counter
                    counter = 0
                i += 1
                if i == len(p) and counter > 1:
                    consecutive_counter += counter
        if len(self.__joined_dot_split_suffix_free) == 0:
            return [0]
        else:
            return [consecutive_counter / len(self.__joined_dot_split_suffix_free)]
    
    
    
    def extract_all_features(self, data):
        """
        Function extracting all available features to a np feature array.
        :param data: iterable containing domain name strings
        :return: feature matrix as np array
        """
        feature_matrix = [self.extract_features(d, self.ALL_FEATURES) for d in data ]
    
        return np.array(feature_matrix)
    
    def extract_all_features_parallel(self, data, n_jobs=-1):
        """
        Function extracting all available features to a np feature array.
        :param data: iterable containing domain name strings
        :return: feature matrix as np array
        """
        parallel = Parallel(n_jobs=n_jobs, verbose=1)
        feature_matrix = parallel(
            delayed(self.extract_features)(d, self.ALL_FEATURES)
            for d in data
            )
    
        return np.array(feature_matrix)
    
    
    def extract_all_features_single(self, d):
        """
        Extracts all features of a single domain name
        :param d: string, domain name
        :return: extracted features as np array
        """
        return np.array(self.extract_features(d, self.ALL_FEATURES))
    
