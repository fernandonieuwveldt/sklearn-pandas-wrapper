
class FeatureSelectionTransformerWrapper:
    """
    The feature selector class of transformer reduces the number of features by 
    selecting important features based on some criteria.

    These transformers have a get_support() method that will used to return the
    of the data frame
    """
    pass


class PreProcessingTransformerWrapper:
    """
    Some of the preprocessing transformer increases the number of features based on the encoding.
    These ones have get_feature_names().
    """
    pass


class DecompositionTransformerWrapper:
    """
    Decomposition class of transformers will not be handled
    """
    pass


class CrossDecompositionTransformerWrapper:
    """
    Decomposition class of transformers will not be handled
    """
    pass


class FeatureExtractionTransformerWrapper:
    """
    Some of the FeatureExtraction transformers increases the number of features based on the encoding.
    These ones have get_feature_names().
    """
    pass


class TransformerWrapperDict:
    """
    Helper class to map type of transformer to the appropriate wrapper class
    """
    _TRANSFORMER_DICT = {'feature_selection': FeatureSelectionTransformerWrapper,
                         'preprocessing': PreProcessingTransformerWrapper,
                         'decomposition': DecompositionTransformerWrapper,
                         'cross_decomposition': CrossDecompositionTransformerWrapper}

