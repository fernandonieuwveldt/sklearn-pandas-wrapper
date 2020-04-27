import sklearn
import sklearn.pipeline
import pandas
import numpy


def _wrap_transformer(transformer_list=None):
    """
    Wrapper function for steps in Pipeline or FeatureUnion
    """
    wrapped_transformers = []
    for name, transformer in transformer_list:
       if sklearn.base.is_classifier(transformer) or sklearn.base.is_regressor(transformer):
           wrapped_step = (name, transformer)
       elif type(transformer) == sklearn.pipeline.Pipeline:
           wrapped_step = (name, PandasPipelineWrapper(transformer))
       else:
           wrapped_step = (name, PandasTransformerWrapper(transformer))
       wrapped_transformers.append(wrapped_step)
    return wrapped_transformers


class BaseTransformerWrapper:
    """
    Base wrapper for all transformers
    """
    _MODULES_NOT_IMPLEMENTED = ['decomposition', 'cross_decomposition']
    def __init__(self, base_transformer_object):
        self._validate_transformer(base_transformer_object)
        self.__class__ = type(base_transformer_object.__class__.__name__,
                              (self.__class__, base_transformer_object.__class__),
                              {})
        self.__dict__ = base_transformer_object.__dict__
        self.module = type(base_transformer_object).__dict__['__module__'].split('.')[1]
        self.is_sparse = self.__dict__.get('sparse', False)

    def _validate_transformer(self, transformer=None):
        """
        Check if transformer is a valid sklearn transformer
        """
        is_transformer = all(map(lambda method: hasattr(transformer, method), ['fit', 'transform', 'fit_transform']))
        if not is_transformer:
            raise ValueError('Not a valid transformer')

    def _check_input_type(self, data_frame=None):
        """
        validate input data
        """
        if isinstance(data_frame, pandas.core.frame.DataFrame):
            return True
        else:
            raise ValueError('Input should be a pandas data frame ')

    def _get_feature_names(self):
        """
        check type of transformer and return feature names if applicable
        """
        if self.module in self._MODULES_NOT_IMPLEMENTED:
            return []
        if hasattr(self, 'get_feature_names'):
            return self.get_feature_names(self.feature_names)
        if hasattr(self, 'get_support'):
            return self.feature_names[self.get_support()]
        return self.feature_names

class PandasTransformerWrapper(BaseTransformerWrapper):
    """
    Wrap sklearn Transformer return type from numpy to pandas with column names
    """
    def fit(self, data_frame=None, y=None):
        """
        Fit the specified transformer and set column names attribute
        """
        # feature names will be used for transform output
        self._check_input_type(data_frame)
        self.feature_names = data_frame.columns
        super(BaseTransformerWrapper, self).fit(data_frame, y)
        return self

    def transform(self, data_frame=None, y=None):
        """
        Apply transformer and cast output as a Pandas DataFrame
        """
        self._check_input_type(data_frame)
        feature_names = self._get_feature_names()
        data_frame_transformed = super(BaseTransformerWrapper, self).transform(data_frame) 
        # check sparsity: output need to be dense array not sparse
        if self.is_sparse:
            data_frame_transformed = data_frame_transformed.toarray()
        if any(feature_names):
            return pandas.DataFrame(data_frame_transformed, columns=feature_names) 
        else:
            return data_frame_transformed


class PandasPipelineWrapper(sklearn.pipeline.Pipeline):
    """
    Wrap sklearn Pipeline steps with PandasTransformerWrapper
    """
    def __init__(self, steps, **kwargs):
        super().__init__(steps=_wrap_transformer(steps),
                         **kwargs)


class PandasFeatureUnionWrapper(sklearn.pipeline.FeatureUnion):
    """
    Wrap FeatureUnion to persist feature names through pipeline
    
    """
    def __init__(self, transformer_list, **kwargs):
        super().__init__(transformer_list=_wrap_transformer(transformer_list),
                         **kwargs)

    def _get_feature_names(self):
        """
        Get feature names of all transformers and concatenate
        """
        feature_names = []
        for name, transformer in self.transformer_list:
            feature_names.append(transformer._get_feature_names())
        return numpy.concatenate(feature_names, axis=0)

    def transform(self, data_frame=None):
        """
        Apply transformer and return data frame with feature names
        """
        feature_names = self._get_feature_names()
        data_frame_transformed = super().transform(data_frame)
        if any(feature_names):
            return pandas.DataFrame(data_frame_transformed, columns=feature_names) 
        else:
            return data_frame_transformed
