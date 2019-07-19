""" Implements instance extraction and decoding for antecedent trees.

This module implements antecedent trees (Fernandes et al., 2014) within a
framework that expresses coreference resolution as predicting latent structures,
while performing learning using a latent structured perceptron with
cost-augmented inference.

Hence, antecedent trees are expressed as as predicting a latent graph.
In particular, let m_1, ..., m_n be all mentions in a document. Let m_0 be a
dummy mention for anaphoricity determination. We predict
the graph with nodes m_0, ..., m_n and with arcs (m_j, m_i) which correspond to
antecedent decisions. In particular, for each j there exists exactly one i < j
such that (m_j, m_i) is in the graph. Such a graph is called aa *substructure*
(for antecedent trees, substructures and structures coincide).

To implement antecedent trees, this module contains a function that defines the
search space for the graphs, and a decoder that computes the best-scoring tree
of antecedent decisions, and the best-scoring tree of antecedent decisions
consistent with the gold annotation (i.e. only having pairs of coreferent
mentions as arcs).

Reference:

    - Eraldo Fernandes, Cicero dos Santos, and Ruy Milidiu. 2014. Latent trees
      for coreference resolution. *Computational Linguistics*, 40(4):801-835.
      http://www.aclweb.org/anthology/J14-4004
"""

from __future__ import division


import array
import numpy as np


from cort.coreference import perceptrons


__author__ = 'martscsn'


def get_candidate_pairs(mentions, max_distance=50, max_distance_with_match=500, debug=False):
    '''
    Yield tuples of mentions, dictionnary of candidate antecedents for the mention
    Arg:
        mentions: an iterator over mention indexes (as returned by get_candidate_mentions)
        max_mention_distance : max distance between a mention and its antecedent
        max_mention_distance_string_match : max distance between a mention and
            its antecedent when there is a proper noun match
    '''
    if debug: print("get_candidate_pairs: mentions", mentions)

    if max_distance_with_match is not None:
        word_to_mentions = {}
        for i in range(len(mentions)):
            if mentions[i].is_dummy():
                continue
            for tok in mentions[i].attributes["tokens"]:
                if not tok in word_to_mentions:
                    word_to_mentions[tok] = [i]
                else:
                    word_to_mentions[tok].append(i)

    for i in range(1, len(mentions)):
        antecedents = set([mentions[k] for k in range(i)]) if max_distance is None \
                 else set([mentions[k] for k in range(max(0, i - max_distance), i)])

        antecedents.add(mentions[0])

        if debug: print("antecedents", antecedents)
        if max_distance_with_match is not None:
            for tok in mentions[i].attributes["tokens"]:
                with_string_match = word_to_mentions.get(tok, None)
                for match_idx in with_string_match:
                    if match_idx < i and match_idx >= i - max_distance_with_match:
                        antecedents.add(mentions[match_idx])

        yield i, antecedents


def extract_substructures_limited(doc):
    """ Extract the search space for the antecedent tree model,

    The mention ranking model consists in computing the optimal antecedent for
    each anaphor. These decisions are represented as edges in a tree of
    anaphor-antecedent decisions. This functions extracts the search space for
    the tree.

    The search space is represented as a nested list of mention pairs. The
    mention pairs are candidate arcs in the graph. The nested list contains
    only one list, since antecedent trees have only one substructure for
    each document.

    The list contains all potential (anaphor, antecedent) pairs in the
    following order: (m_1, m_0), (m_2, m_1), (m_2, m_0), (m_3, m_2), ...,
    where m_j is the jth mention in the document.

    Args:
        doc (CoNLLDocument): The document to extract substructures from.

    Returns:
        (list(list(Mention, Mention))): The nested list of mention pairs
        describing the search space for the substructures.
    """
    substructure = []
    for i, antecedents in get_candidate_pairs(doc.system_mentions, 10):
        ana = doc.system_mentions[i]
        for ante in antecedents:
            substructure.append((ana, ante))

    return [substructure]


def extract_substructures(doc):
    """ Extract the search space for the antecedent tree model,

    The mention ranking model consists in computing the optimal antecedent for
    each anaphor. These decisions are represented as edges in a tree of
    anaphor-antecedent decisions. This functions extracts the search space for
    the tree.

    The search space is represented as a nested list of mention pairs. The
    mention pairs are candidate arcs in the graph. The nested list contains
    only one list, since antecedent trees have only one substructure for
    each document.

    The list contains all potential (anaphor, antecedent) pairs in the
    following order: (m_1, m_0), (m_2, m_1), (m_2, m_0), (m_3, m_2), ...,
    where m_j is the jth mention in the document.

    Args:
        doc (CoNLLDocument): The document to extract substructures from.

    Returns:
        (list(list(Mention, Mention))): The nested list of mention pairs
        describing the search space for the substructures.
    """
    substructure = []

    # iterate over mentions
    for i, ana in enumerate(doc.system_mentions):

        # iterate in reversed order over candidate antecedents
        for ante in sorted(doc.system_mentions[:i], reverse=True):
            substructure.append((ana, ante))

    return [substructure]


class AntecedentTreePerceptron(perceptrons.Perceptron):

    def my_score_arc(self,
                      prior,
                      weights,
                      cost_scaling,
                      costs,
                      nonnumeric_features,
                      numeric_features,
                      numeric_vals):

        score = 0.0

        score += prior
        score += cost_scaling * costs

        for index in range(nonnumeric_features.shape[0]):
            score += weights[nonnumeric_features[index]]

        for index in range(numeric_features.shape[0]):
            score += weights[numeric_features[index]]*numeric_vals[index]

        return score

    def score_arc(self, arc, arc_information, label="+"):
        """ Score an arc according to priors, weights and costs.

        Args:
            arc ((Mention, Mention)): The pair of mentions constituting the arc.
            arc_information (dict((Mention, Mention),
                                  ((array, array, array), list(int), bool)):
                A mapping of arcs (= mention pairs) to information about these
                arcs. The information consists of the features, the costs for
                the arc (for each label), and whether predicting the arc to be
                coreferent is consistent with the gold annotation). The features
                are divided in three arrays: the first array contains the non-
                numeric features, the second array the numeric features, and the
                third array the values for the numeric features. The features
                are represented as integers via feature hashing.
            label (str): The label of the arc. Defaults to "+".

        Returns:
            float: The sum of all weights for the features, plus the scaled
                costs for predicting the arc, plus the prior for the label.
        """

        features, costs, consistent = arc_information[arc]

        nonnumeric_features, numeric_features, numeric_vals = features

        return self._my_score_arc_c(
            costs,
            nonnumeric_features,
            numeric_features,
            numeric_vals
        )

    def find_best_arcs(self, arcs, arc_information, label="+"):
        """ Find the highest-scoring arc and arc consistent with the gold
        information among a set of arcs.

        Args:
            arcs (list((Mention, Mention)): A list of mention pairs constituting
                arcs.
            arc_information (dict((Mention, Mention),
                                  ((array, array, array), list(int), bool)):
                A mapping of arcs (= mention pairs) to information about these
                arcs. The information consists of the features, the costs for
                the arc (for each label), and whether predicting the arc to be
                coreferent is consistent with the gold annotation). The features
                are divided in three arrays: the first array contains the non-
                numeric features, the second array the numeric features, and the
                third array the values for the numeric features. The features
                are represented as integers via feature hashing.
            label (str): The label of the arcs. Defaults to "+".

        Returns:
            A 5-tuple describing the highest-scoring anaphor-antecedent
            decision, and the highest-scoring anaphor-antecedent decision
            consistent with the gold annotation. The tuple consists of:

                - **best** (*(Mention, Mention)*): the highest-scoring
                  anaphor-antecedent decision.
                - **max_val** (*float*): the score of the highest-scoring
                  anaphor-antecedent decision,
                - **best_cons** (*(Mention, Mention)*): the highest-scoring
                  anaphor-antecedent decision consistent with the gold
                  annotation.
                - **max_const** (*float*): the score of the highest-scoring
                  anaphor-antecedent decision consistent with the gold
                  annotation.
                - **is_consistent** (*bool*): whether the highest-scoring
                  anaphor-antecedent decision is consistent with the gold
                  information.
        """
        max_val = float("-inf")
        best = None

        max_cons = float("-inf")
        best_cons = None

        best_is_consistent = False

        for arc in arcs:
            features, costs, consistent = arc_information[arc]
            score = self.score_arc(arc, arc_information)

            if score > max_val:
                best = arc
                max_val = score
                best_is_consistent = consistent

            if score > max_cons and consistent:
                best_cons = arc
                max_cons = score

        # print(f"max_val:{max_val}, max_cons:{max_cons}")
        return best, max_val, best_cons, max_cons, best_is_consistent

    def argmax(self, substructure, arc_information):
        return self.argmax_general(substructure, arc_information)

    """ A perceptron for antecedent trees. """
    def argmax_strict(self, substructure, arc_information):
        """ Decoder for antecedent trees.

        Compute highest-scoring antecedent tree and highest-scoring antecedent
        tree consistent with the gold annotation.

        Args:
            substructure (list((Mention, Mention))): The list of mention pairs
                which define the search space for one substructure. For mention
                ranking, this list contains all potential anaphor-antecedent
                pairs in the following order:
                (m_1, m_0), (m_2, m_1), (m_2, m_0), (m_3, m_2), ...
            arc_information (dict((Mention, Mention),
                                  ((array, array, array), list(int), bool)):
                A mapping of arcs (= mention pairs) to information about these
                arcs. The information consists of the features, the costs for
                the arc (for each label), and whether predicting the arc to be
                coreferent is consistent with the gold annotation). The features
                are divided in three arrays: the first array contains the non-
                numeric features, the second array the numeric features, and the
                third array the values for the numeric features. The features
                are represented as integers via feature hashing.

        Returns:
            A 7-tuple describing the highest-scoring antecedent tree, and the
            highest-scoring antecedent tree consistent with the gold
            annotation. The tuple consists of:

                - **best_arcs** (*list((Mention, Mention))*): the arcs
                  constituting the highest-scoring antecedent tree,
                - **best_labels** (*list(str)*): empty, the antecedent tree
                  approach does not employ any labels,
                - **best_scores** (*list(float)*): the scores of the
                  arcs in the highest-scoring antecedent tree,
                - **best_cons_arcs** (*list((Mention, Mention))*): the arcs
                  constituting the highest-scoring antecedent tree consistent
                  with the gold annotation.
                - **best_cons_labels** (*list(str)*): empty, the antecedent
                  tree approach does not employ any labels
                - **best_cons_scores** (*list(float)*): the scores of the
                  arcs in the highest-scoring antecedent tree consistent with
                  the gold annotation,
                - **is_consistent** (*bool*): whether the highest-scoring
                  antecedent tree is consistent with the gold annotation.
        """
        if not substructure:
            return [], [], [], [], [], [], True

        number_mentions = len(substructure[0][0].document.system_mentions)

        arcs = []
        arcs_scores = []
        coref_arcs = []
        coref_arcs_scores = []

        is_consistent = True

        for ana_index in range(1, number_mentions):

            first_arc = ana_index*(ana_index-1)//2
            last_arc = first_arc + ana_index

            best, max_val, best_cons, max_cons, best_is_consistent = \
                self.find_best_arcs(substructure[first_arc:last_arc],
                                    arc_information)

            if best is not None:
                arcs.append(best)
                arcs_scores.append(max_val)
            else:
                print("No best")

            if best_cons is not None:
                coref_arcs.append(best_cons)
                coref_arcs_scores.append(max_cons)
            else:
                pass #print("No best")

            is_consistent &= best_is_consistent

        #print("Done argmax")

        return (
            arcs,
            [],
            arcs_scores,
            coref_arcs,
            [],
            coref_arcs_scores,
            is_consistent
        )

    """ A perceptron for antecedent trees. """
    def argmax_general(self, substructure, arc_information):
        """ Decoder for antecedent trees.

        Compute highest-scoring antecedent tree and highest-scoring antecedent
        tree consistent with the gold annotation.

        Args:
            substructure (list((Mention, Mention))): The list of mention pairs
                which define the search space for one substructure. For mention
                ranking, this list contains all potential anaphor-antecedent
                pairs in the following order:
                (m_1, m_0), (m_2, m_1), (m_2, m_0), (m_3, m_2), ...
            arc_information (dict((Mention, Mention),
                                  ((array, array, array), list(int), bool)):
                A mapping of arcs (= mention pairs) to information about these
                arcs. The information consists of the features, the costs for
                the arc (for each label), and whether predicting the arc to be
                coreferent is consistent with the gold annotation). The features
                are divided in three arrays: the first array contains the non-
                numeric features, the second array the numeric features, and the
                third array the values for the numeric features. The features
                are represented as integers via feature hashing.

        Returns:
            A 7-tuple describing the highest-scoring antecedent tree, and the
            highest-scoring antecedent tree consistent with the gold
            annotation. The tuple consists of:

                - **best_arcs** (*list((Mention, Mention))*): the arcs
                  constituting the highest-scoring antecedent tree,
                - **best_labels** (*list(str)*): empty, the antecedent tree
                  approach does not employ any labels,
                - **best_scores** (*list(float)*): the scores of the
                  arcs in the highest-scoring antecedent tree,
                - **best_cons_arcs** (*list((Mention, Mention))*): the arcs
                  constituting the highest-scoring antecedent tree consistent
                  with the gold annotation.
                - **best_cons_labels** (*list(str)*): empty, the antecedent
                  tree approach does not employ any labels
                - **best_cons_scores** (*list(float)*): the scores of the
                  arcs in the highest-scoring antecedent tree consistent with
                  the gold annotation,
                - **is_consistent** (*bool*): whether the highest-scoring
                  antecedent tree is consistent with the gold annotation.
        """
        if not substructure:
            return [], [], [], [], [], [], True

        number_mentions = len(substructure[0][0].document.system_mentions)

        arcs = []
        arcs_scores = []
        coref_arcs = []
        coref_arcs_scores = []

        is_consistent = True

        first_arc = 0
        ana = substructure[0][0]
        consistent_counter = 0
        total_consistent = 0
        for i in range(len(substructure)):
            last_arc = i

            _, _, consistent = arc_information[substructure[i]]
            total_consistent += consistent

            if substructure[i][0] != ana or i == len(substructure) - 1:
                if i == len(substructure) - 1:
                    last_arc += 1

                best, max_val, best_cons, max_cons, best_is_consistent = \
                    self.find_best_arcs(substructure[first_arc:last_arc],
                                        arc_information)

                if best is not None:
                    arcs.append(best)
                    arcs_scores.append(max_val)
                else:
                    print("No best")

                if best_cons is not None:
                    coref_arcs.append(best_cons)
                    coref_arcs_scores.append(max_cons)
                else:
                    pass  # print("No best")

                is_consistent &= best_is_consistent
                consistent_counter += best_is_consistent

                first_arc = i
                ana = substructure[first_arc][0]

        doc_id = substructure[0][0].document.identifier
        print(f"Done argmax {consistent_counter}/{total_consistent} \t{doc_id}")
        # for arc in arcs:
        #    print(f"Arc:{arc}")

        return (
            arcs,
            [],
            arcs_scores,
            coref_arcs,
            [],
            coref_arcs_scores,
            is_consistent
        )
