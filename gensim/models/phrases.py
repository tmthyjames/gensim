#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""Automatically detect common phrases -- multi-word expressions / word n-grams -- from a stream of sentences.

Inspired by:

* `Mikolov, et. al: "Distributed Representations of Words and Phrases and their Compositionality"
  <https://arxiv.org/abs/1310.4546>`_
* `"Normalized (Pointwise) Mutual Information in Colocation Extraction" by Gerlof Bouma
  <https://svn.spraakdata.gu.se/repos/gerlof/pub/www/Docs/npmi-pfd.pdf>`_


Examples
--------
>>> from gensim.test.utils import datapath
>>> from gensim.models.word2vec import Text8Corpus
>>> from gensim.models.phrases import Phrases, Phraser
>>>
>>> sentences = Text8Corpus(datapath('testcorpus.txt'))
>>> phrases = Phrases(sentences, min_count=1, threshold=1)  # train model
>>> phrases[[u'trees', u'graph', u'minors']]  # apply model to sentence
[u'trees_graph', u'minors']
>>>
>>> phrases.add_vocab([["hello", "world"], ["meow"]])  # update model with new sentences
>>>
>>> bigram = Phraser(phrases)  # construct faster model (this is only an wrapper)
>>> bigram[[u'trees', u'graph', u'minors']]  # apply model to sentence
[u'trees_graph', u'minors']
>>>
>>> for sent in bigram[sentences]:  # apply model to text corpus
...     pass

"""

import sys
import os
import logging
import warnings
from collections import defaultdict
import functools as ft
import itertools as it
from math import log
import pickle
import six

from six import iteritems, string_types, PY2, next

from gensim import utils, interfaces
from gensim.models.doc2vec import TaggedDocument

if PY2:
    from inspect import getargspec
else:
    from inspect import getfullargspec as getargspec

logger = logging.getLogger(__name__)


def _is_single(obj):
    """Check whether `obj` is a single document or an entire corpus.

    Parameters
    ----------
    obj : object

    Return
    ------
    (bool, object)
        (is_single, new) tuple, where `new` yields the same sequence as `obj`.

    Notes
    -----
    `obj` is a single document if it is an iterable of strings. It is a corpus if it is an iterable of documents.

    """
    obj_iter = iter(obj)
    temp_iter = obj_iter
    try:
        peek = next(obj_iter)
        obj_iter = it.chain([peek], obj_iter)
    except StopIteration:
        # An empty object is a single document
        return True, obj
    if isinstance(peek, string_types):
        # It's a document, return the iterator
        return True, obj_iter
    if temp_iter is obj:
        # Checking for iterator to the object
        return False, obj_iter
    else:
        # If the first item isn't a string, assume obj is a corpus
        return False, obj


class SentenceAnalyzer(object):
    """Base util class for :class:`~gensim.models.phrases.Phrases` and :class:`~gensim.models.phrases.Phraser`."""
    def score_item(self, worda, wordb, components, scorer):
        """Get bi-gram score statistics.

        Parameters
        ----------
        worda : str
            First word of bi-gram.
        wordb : str
            Second word of bi-gram.
        components : generator
            Contain all phrases.
        scorer : function
            Scorer function, as given to :class:`~gensim.models.phrases.Phrases`.
            See :func:`~gensim.models.phrases.npmi_scorer` and :func:`~gensim.models.phrases.original_scorer`.

        Returns
        -------
        float
            Score for given bi-gram. If bi-gram not present in dictionary - return -1.

        """
        vocab = self.vocab
        if worda in vocab and wordb in vocab:
            bigram = self.delimiter.join(components)
            if bigram in vocab:
                return scorer(
                    worda_count=float(vocab[worda]),
                    wordb_count=float(vocab[wordb]),
                    bigram_count=float(vocab[bigram]))
        return -1

    def analyze_sentence(self, sentence, threshold, common_terms, scorer):
        """Analyze a sentence, detecting any bigrams that should be concatenated.

        Parameters
        ----------
        sentence : iterable of str
            Token sequence representing the sentence to be analyzed.
        threshold : float
            The minimum score for a bigram to be taken into account.
        common_terms : list of object
            List of common terms, they receive special treatment.
        scorer : function
            Scorer function, as given to :class:`~gensim.models.phrases.Phrases`.
            See :func:`~gensim.models.phrases.npmi_scorer` and :func:`~gensim.models.phrases.original_scorer`.

        Yields
        ------
        (str, score)
            If bi-gram detected, a tuple where the first element is a detect bigram, second its score.
            Otherwise, the first tuple element is a single word and second is None.

        """
        s = [utils.any2utf8(w) for w in sentence]
        # adding None is a trick that helps getting an automatic happy ending
        # as it won't be a common_word, nor score
        s.append(None)
        last_uncommon = None
        in_between = []
        for word in s:
            is_common = word in common_terms
            if not is_common and last_uncommon:
                chain = [last_uncommon] + in_between + [word]
                # test between last_uncommon
                score = self.score_item(
                    worda=last_uncommon,
                    wordb=word,
                    components=chain,
                    scorer=scorer,
                )
                if score > threshold:
                    yield (chain, score)
                    last_uncommon = None
                    in_between = []
                else:
                    # release words individually
                    for w in it.chain([last_uncommon], in_between):
                        yield (w, None)
                    in_between = []
                    last_uncommon = word
            elif not is_common:
                last_uncommon = word
            else:  # common term
                if last_uncommon:
                    # wait for uncommon resolution
                    in_between.append(word)
                else:
                    yield (word, None)


class PhrasesTransformation(interfaces.TransformationABC):
    """Base util class for :class:`~gensim.models.phrases.Phrases` and :class:`~gensim.models.phrases.Phraser`."""

    @classmethod
    def load(cls, *args, **kwargs):
        """Load a previously saved :class:`~gensim.models.phrases.Phrases` /
        :class:`~gensim.models.phrases.Phraser` class. Handles backwards compatibility from older
        :class:`~gensim.models.phrases.Phrases` / :class:`~gensim.models.phrases.Phraser`
        versions which did not support pluggable scoring functions.

        Parameters
        ----------
        args : object
            Sequence of arguments, see :class:`~gensim.utils.SaveLoad.load` for more information.
        kwargs : object
            Sequence of arguments, see :class:`~gensim.utils.SaveLoad.load` for more information.

        """
        model = super(PhrasesTransformation, cls).load(*args, **kwargs)
        # update older models
        # if no scoring parameter, use default scoring
        if not hasattr(model, 'scoring'):
            logger.info('older version of %s loaded without scoring function', cls.__name__)
            logger.info('setting pluggable scoring method to original_scorer for compatibility')
            model.scoring = original_scorer
        # if there is a scoring parameter, and it's a text value, load the proper scoring function
        if hasattr(model, 'scoring'):
            if isinstance(model.scoring, six.string_types):
                if model.scoring == 'default':
                    logger.info('older version of %s loaded with "default" scoring parameter', cls.__name__)
                    logger.info('setting scoring method to original_scorer pluggable scoring method for compatibility')
                    model.scoring = original_scorer
                elif model.scoring == 'npmi':
                    logger.info('older version of %s loaded with "npmi" scoring parameter', cls.__name__)
                    logger.info('setting scoring method to npmi_scorer pluggable scoring method for compatibility')
                    model.scoring = npmi_scorer
                else:
                    raise ValueError(
                        'failed to load %s model with unknown scoring setting %s' % (cls.__name__, model.scoring))
        # if there is no common_terms attribute, initialize
        if not hasattr(model, "common_terms"):
            logger.info('older version of %s loaded without common_terms attribute', cls.__name__)
            logger.info('setting common_terms to empty set')
            model.common_terms = frozenset()
        return model


class Phrases(interfaces.TransformationABC):
    """
    Detect phrases, based on collected collocation counts. Adjacent words that appear
    together more frequently than expected are joined together with the `_` character.
    It can be used to generate phrases on the fly, using the `phrases[sentence]`
    and `phrases[corpus]` syntax.
    """
    def __init__(self, sentences=None, min_count=5, threshold=10.0,
                 max_vocab_size=40000000, delimiter=b'_', progress_per=10000,
                 doc2vec=False):
        """
        Initialize the model from an iterable of `sentences`. Each sentence must be
        a list of words (unicode strings) that will be used for training.
        The `sentences` iterable can be simply a list, but for larger corpora,
        consider a generator that streams the sentences directly from disk/network,
        without storing everything in RAM. See :class:`BrownCorpus`,
        :class:`Text8Corpus` or :class:`LineSentence` in the :mod:`gensim.models.word2vec`
        module for such examples.
        `min_count` ignore all words and bigrams with total collected count lower
        than this.
        `threshold` represents a threshold for forming the phrases (higher means
        fewer phrases). A phrase of words `a` and `b` is accepted if
        `(cnt(a, b) - min_count) * N / (cnt(a) * cnt(b)) > threshold`, where `N` is the
        total vocabulary size.
        `max_vocab_size` is the maximum size of the vocabulary. Used to control
        pruning of less common words, to keep memory under control. The default
        of 40M needs about 3.6GB of RAM; increase/decrease `max_vocab_size` depending
        on how much available memory you have.
        `delimiter` is the glue character used to join collocation tokens, and
        should be a byte string (e.g. b'_').
        """
        if min_count <= 0:
            raise ValueError("min_count should be at least 1")

        if threshold <= 0:
            raise ValueError("threshold should be positive")

        self.min_count = min_count
        self.threshold = threshold
        self.max_vocab_size = max_vocab_size
        self.vocab = defaultdict(int)  # mapping between utf8 token => its count
        self.min_reduce = 1  # ignore any tokens with count smaller than this
        self.delimiter = delimiter
        self.progress_per = progress_per
        self.doc2vec = doc2vec

        if sentences is not None:
            self.add_vocab(sentences)

    def __str__(self):
        """Get short string representation of this phrase detector."""
        return "%s<%i vocab, min_count=%s, threshold=%s, max_vocab_size=%s>" % (
            self.__class__.__name__, len(self.vocab), self.min_count,
            self.threshold, self.max_vocab_size)

    @staticmethod
    def learn_vocab(sentences, max_vocab_size, delimiter=b'_', progress_per=10000, doc2vec=False):
        """Collect unigram/bigram counts from the `sentences` iterable."""
        sentence_no = -1
        total_words = 0
        logger.info("collecting all words and their counts")
        vocab = defaultdict(int)
        min_reduce = 1
        for sentence_no, sentence in enumerate(sentences):
                
            if sentence_no % progress_per == 0:
                logger.info("PROGRESS: at sentence #%i, processed %i words and %i word types" %
                            (sentence_no, total_words, len(vocab)))
                
            ######################################### ngram-paragraph vectors
            if doc2vec:
                sentence = sentence.words
            
            sentence = [utils.any2utf8(w) for w in sentence]
                
            for bigram in zip(sentence, sentence[1:]):
                vocab[bigram[0]] += 1
                vocab[delimiter.join(bigram)] += 1
                total_words += 1

            if sentence:  # add last word skipped by previous loop
                word = sentence[-1]
                vocab[word] += 1

            if len(vocab) > max_vocab_size:
                utils.prune_vocab(vocab, min_reduce)
                min_reduce += 1

        logger.info("collected %i word types from a corpus of %i words (unigram + bigrams) and %i sentences" %
                    (len(vocab), total_words, sentence_no + 1))
        return min_reduce, vocab

    def add_vocab(self, sentences):
        """
        Merge the collected counts `vocab` into this phrase detector.
        """
        # uses a separate vocab to collect the token counts from `sentences`.
        # this consumes more RAM than merging new sentences into `self.vocab`
        # directly, but gives the new sentences a fighting chance to collect
        # sufficient counts, before being pruned out by the (large) accummulated
        # counts collected in previous learn_vocab runs.
        min_reduce, vocab = self.learn_vocab(sentences, self.max_vocab_size, 
                                             self.delimiter, self.progress_per, self.doc2vec)

        if len(self.vocab) > 0:
            logger.info("merging %i counts into %s", len(vocab), self)
            self.min_reduce = max(self.min_reduce, min_reduce)
            for word, count in iteritems(vocab):
                self.vocab[word] += count
            if len(self.vocab) > self.max_vocab_size:
                utils.prune_vocab(self.vocab, self.min_reduce)
                self.min_reduce += 1
            logger.info("merged %s", self)
        else:
            # in common case, avoid doubling gigantic dict
            logger.info("using %i counts as vocab in %s", len(vocab), self)
            self.vocab = vocab

    def export_phrases(self, sentences, out_delimiter=b' ', as_tuples=False):
        """
        Generate an iterator that contains all phrases in given 'sentences'
        Example::
          >>> sentences = Text8Corpus(path_to_corpus)
          >>> bigram = Phrases(sentences, min_count=5, threshold=100)
          >>> for phrase, score in bigram.export_phrases(sentences):
          ...     print(u'{0}\t{1}'.format(phrase, score))
            then you can debug the threshold with generated tsv
        """
        for sentence in sentences:
            s = [utils.any2utf8(w) for w in sentence]
            last_bigram = False
            vocab = self.vocab
            threshold = self.threshold
            delimiter = self.delimiter  # delimiter used for lookup
            min_count = self.min_count
            for word_a, word_b in zip(s, s[1:]):
                if word_a in vocab and word_b in vocab:
                    bigram_word = delimiter.join((word_a, word_b))
                    if bigram_word in vocab and not last_bigram:
                        pa = float(vocab[word_a])
                        pb = float(vocab[word_b])
                        pab = float(vocab[bigram_word])
                        score = (pab - min_count) / pa / pb * len(vocab)
                        # logger.debug("score for %s: (pab=%s - min_count=%s) / pa=%s / pb=%s * vocab_size=%s = %s",
                        #     bigram_word, pab, self.min_count, pa, pb, len(self.vocab), score)
                        if score > threshold:
                            if as_tuples:
                                yield ((word_a, word_b), score)
                            else:
                                yield (out_delimiter.join((word_a, word_b)), score)
                            last_bigram = True
                            continue
                        last_bigram = False

    def __getitem__(self, sentence):
        """
        Convert the input tokens `sentence` (=list of unicode strings) into phrase
        tokens (=list of unicode strings, where detected phrases are joined by u'_').
        If `sentence` is an entire corpus (iterable of sentences rather than a single
        sentence), return an iterable that converts each of the corpus' sentences
        into phrases on the fly, one after another.
        Example::
          >>> sentences = Text8Corpus(path_to_corpus)
          >>> bigram = Phrases(sentences, min_count=5, threshold=100)
          >>> for sentence in phrases[sentences]:
          ...     print(u' '.join(s))
            he refuted nechaev other anarchists sometimes identified as pacifist anarchists advocated complete
            nonviolence leo_tolstoy
        """
        warnings.warn("For a faster implementation, use the gensim.models.phrases.Phraser class")
        try:
            is_single = not sentence or isinstance(sentence[0], string_types) or self.doc2vec
        except:
            is_single = False
        if not is_single:
            # if the input is an entire corpus (rather than a single sentence),
            # return an iterable stream.
            return self._apply(sentence)
        
        ######################################### ngram-paragraph vectors
        if self.doc2vec:
            tag = sentence.tags
            sentence = sentence.words

        s, new_s = [utils.any2utf8(w) for w in sentence], []
        
        last_bigram = False
        vocab = self.vocab
        threshold = self.threshold
        delimiter = self.delimiter
        min_count = self.min_count
        for word_a, word_b in zip(s, s[1:]):
            if word_a in vocab and word_b in vocab:
                bigram_word = delimiter.join((word_a, word_b))
                if bigram_word in vocab and not last_bigram:
                    pa = float(vocab[word_a])
                    pb = float(vocab[word_b])
                    pab = float(vocab[bigram_word])
                    score = (pab - min_count) / pa / pb * len(vocab)
                    # logger.debug("score for %s: (pab=%s - min_count=%s) / pa=%s / pb=%s * vocab_size=%s = %s",
                    #     bigram_word, pab, self.min_count, pa, pb, len(self.vocab), score)
                    if score > threshold:
                        new_s.append(bigram_word)
                        last_bigram = True
                        continue

            if not last_bigram:
                new_s.append(word_a)
            last_bigram = False

        if s:  # add last word skipped by previous loop
            last_token = s[-1]
            if not last_bigram:
                new_s.append(last_token)
                
        ######################################### ngram-paragraph vectors
        if self.doc2vec:
            return TaggedDocument(words=[utils.to_unicode(w) for w in new_s], tags=tag)
        
        return [utils.to_unicode(w) for w in new_s]

def original_scorer(worda_count, wordb_count, bigram_count, len_vocab, min_count, corpus_word_count):
    """Bigram scoring function, based on the original `Mikolov, et. al: "Distributed Representations
    of Words and Phrases and their Compositionality" <https://arxiv.org/abs/1310.4546>`_.

    Parameters
    ----------
    worda_count : int
        Number of occurrences for first word.
    wordb_count : int
        Number of occurrences for second word.
    bigram_count : int
        Number of co-occurrences for phrase "worda_wordb".
    len_vocab : int
        Size of vocabulary.
    min_count: int
        Minimum collocation count threshold.
    corpus_word_count : int
        Not used in this particular scoring technique.

    Notes
    -----
    Formula: :math:`\\frac{(bigram\_count - min\_count) * len\_vocab }{ (worda\_count * wordb\_count)}`.

    """
    return (bigram_count - min_count) / worda_count / wordb_count * len_vocab


def npmi_scorer(worda_count, wordb_count, bigram_count, len_vocab, min_count, corpus_word_count):
    """Calculation NPMI score based on `"Normalized (Pointwise) Mutual Information in Colocation Extraction"
    by Gerlof Bouma <https://svn.spraakdata.gu.se/repos/gerlof/pub/www/Docs/npmi-pfd.pdf>`_.

    Parameters
    ----------
    worda_count : int
        Number of occurrences for first word.
    wordb_count : int
        Number of occurrences for second word.
    bigram_count : int
        Number of co-occurrences for phrase "worda_wordb".
    len_vocab : int
        Not used.
    min_count: int
        Ignore all bigrams with total collected count lower than this value.
    corpus_word_count : int
        Total number of words in the corpus.

    Notes
    -----
    Formula: :math:`\\frac{ln(prop(word_a, word_b) / (prop(word_a)*prop(word_b)))}{ -ln(prop(word_a, word_b)}`,
    where :math:`prob(word) = \\frac{word\_count}{corpus\_word\_count}`

    """
    if bigram_count >= min_count:
        pa = worda_count / corpus_word_count
        pb = wordb_count / corpus_word_count
        pab = bigram_count / corpus_word_count
        return log(pab / (pa * pb)) / -log(pab)
    else:
        # Return -infinity to make sure that no phrases will be created
        # from bigrams less frequent than min_count
        return float('-inf')


def pseudocorpus(source_vocab, sep, common_terms=frozenset()):
    """Feeds `source_vocab`'s compound keys back to it, to discover phrases.

    Parameters
    ----------
    source_vocab : iterable of list of str
        Tokens vocabulary.
    sep : str
        Separator element.
    common_terms : set, optional
        Immutable set of stopwords.

    Yields
    ------
    list of str
        Phrase.

    """
    for k in source_vocab:
        if sep not in k:
            continue
        unigrams = k.split(sep)
        for i in range(1, len(unigrams)):
            if unigrams[i - 1] not in common_terms:
                # do not join common terms
                cterms = list(it.takewhile(lambda w: w in common_terms, unigrams[i:]))
                tail = unigrams[i + len(cterms):]
                components = [sep.join(unigrams[:i])] + cterms
                if tail:
                    components.append(sep.join(tail))
                yield components


class Phraser(SentenceAnalyzer, PhrasesTransformation):
    """Minimal state & functionality exported from :class:`~gensim.models.phrases.Phrases`.

    The goal of this class is to cut down memory consumption of `Phrases`, by discarding model state
    not strictly needed for the bigram detection task.

    Use this instead of `Phrases` if you do not need to update the bigram statistics with new documents any more.

    """

    def __init__(self, phrases_model):
        """

        Parameters
        ----------
        phrases_model : :class:`~gensim.models.phrases.Phrases`
            Trained phrases instance.

        Notes
        -----
        After the one-time initialization, a :class:`~gensim.models.phrases.Phraser` will be much smaller and somewhat
        faster than using the full :class:`~gensim.models.phrases.Phrases` model.

        Examples
        --------
        >>> from gensim.test.utils import datapath
        >>> from gensim.models.word2vec import Text8Corpus
        >>> from gensim.models.phrases import Phrases, Phraser
        >>>
        >>> sentences = Text8Corpus(datapath('testcorpus.txt'))
        >>> phrases = Phrases(sentences, min_count=1, threshold=1)
        >>>
        >>> bigram = Phraser(phrases)
        >>> sent = [u'trees', u'graph', u'minors']
        >>> print(bigram[sent])
        [u'trees_graph', u'minors']

        """
        self.threshold = phrases_model.threshold
        self.min_count = phrases_model.min_count
        self.delimiter = phrases_model.delimiter
        self.scoring = phrases_model.scoring
        self.common_terms = phrases_model.common_terms
        corpus = self.pseudocorpus(phrases_model)
        self.phrasegrams = {}
        logger.info('source_vocab length %i', len(phrases_model.vocab))
        count = 0
        for bigram, score in phrases_model.export_phrases(corpus, self.delimiter, as_tuples=True):
            if bigram in self.phrasegrams:
                logger.info('Phraser repeat %s', bigram)
            self.phrasegrams[bigram] = (phrases_model.vocab[self.delimiter.join(bigram)], score)
            count += 1
            if not count % 50000:
                logger.info('Phraser added %i phrasegrams', count)
        logger.info('Phraser built with %i phrasegrams', len(self.phrasegrams))

    def pseudocorpus(self, phrases_model):
        """Alias for :func:`gensim.models.phrases.pseudocorpus`.

        Parameters
        ----------
        phrases_model : :class:`~gensim.models.phrases.Phrases`
            Phrases instance.

        Return
        ------
        generator
            Generator with phrases.

        """
        return pseudocorpus(phrases_model.vocab, phrases_model.delimiter, phrases_model.common_terms)

    def score_item(self, worda, wordb, components, scorer):
        """Score a bigram.

        Parameters
        ----------
        worda : str
            First word for comparison.
        wordb : str
            Second word for comparison.
        components : generator
            Contain phrases.
        scorer : {'default', 'npmi'}
            NOT USED.

        Returns
        -------
        float
            Score for given bi-gram, if bi-gram not presented in dictionary - return -1.

        """
        try:
            return self.phrasegrams[tuple(components)][1]
        except KeyError:
            return -1

    def __getitem__(self, sentence):
        """Convert the input sequence of tokens `sentence` into a sequence of tokens where adjacent
        tokens are replaced by a single token if they form a bigram collocation.

        Parameters
        ----------
        sentence : {list of str, iterable of list of str}
            Input sentence or a stream of sentences.

        Return
        ------
        {list of str, iterable of list of str}
            Sentence or sentences with phrase tokens joined by `self.delimiter` character.

        Examples
        ----------
        >>> from gensim.test.utils import datapath
        >>> from gensim.models.word2vec import Text8Corpus
        >>> from gensim.models.phrases import Phrases, Phraser
        >>>
        >>> sentences = Text8Corpus(datapath('testcorpus.txt'))  # Read corpus
        >>>
        >>> phrases = Phrases(sentences, min_count=1, threshold=1) # Train model
        >>> # Create a Phraser object to transform any sentence and turn 2 suitable tokens into 1 phrase
        >>> phraser_model = Phraser(phrases)
        >>>
        >>> sent = [u'trees', u'graph', u'minors']
        >>> print(phraser_model[sent])
        [u'trees_graph', u'minors']
        >>> sent = [[u'trees', u'graph', u'minors'],[u'graph', u'minors']]
        >>> for phrase in phraser_model[sent]:
        ...     print(phrase)
        [u'trees_graph', u'minors']
        [u'graph_minors']

        """
        is_single, sentence = _is_single(sentence)
        if not is_single:
            # if the input is an entire corpus (rather than a single sentence),
            # return an iterable stream.
            return self._apply(sentence)

        delimiter = self.delimiter
        bigrams = self.analyze_sentence(
            sentence,
            threshold=self.threshold,
            common_terms=self.common_terms,
            scorer=None)  # we will use our score_item function redefinition
        new_s = []
        for words, score in bigrams:
            if score is not None:
                words = delimiter.join(words)
            new_s.append(words)
        return [utils.to_unicode(w) for w in new_s]


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s', level=logging.INFO)
    logging.info("running %s", " ".join(sys.argv))

    # check and process cmdline input
    program = os.path.basename(sys.argv[0])
    if len(sys.argv) < 2:
        print(globals()['__doc__'] % locals())
        sys.exit(1)
    infile = sys.argv[1]

    from gensim.models import Phrases  # noqa:F811 for pickle
    from gensim.models.word2vec import Text8Corpus
    sentences = Text8Corpus(infile)

    # test_doc = LineSentence('test/test_data/testcorpus.txt')
    bigram = Phrases(sentences, min_count=5, threshold=100)
    for s in bigram[sentences]:
        print(utils.to_utf8(u' '.join(s)))
