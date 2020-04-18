import sklearn
import sklearn.pipeline
import pandas
import numpy
import pdb

class PandasTransformerWrapper(sklearn.base.TransformerMixin):
    """
    Wrap sklearn Transformer return type from numpy to pandas with column names

    """
    _MODULES_NOT_IMPLEMENTED = ['decomposition', 'cross_decomposition']

    def __init__(self, transformer=None, **kwargs):
        self._validate_transformer(transformer)
        if callable(transformer):
            self.transformer = transformer(**kwargs)
        else:
            self.transformer = transformer
        self.transformer_module = type(self.transformer).__dict__['__module__'].split('.')[1]
        self.is_sparse = self.transformer.get_params().get('sparse', False)

    def _validate_transformer(self, transformer=None):
        """
        Check if transformer is a valid sklearn transformer
        """
        is_transformer = all(map(lambda method: hasattr(transformer, method), ['fit', 'transform', 'fit_transform']))
        if not is_transformer:
            raise ValueError('transformer does not contain all of fit, transform and fit_transform methods')

    def _validate_input(self, data_frame=None):
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
        if self.transformer_module in self._MODULES_NOT_IMPLEMENTED:
            return []
        if hasattr(self.transformer, 'get_feature_names'):
            return self.transformer.get_feature_names()
        if hasattr(self.transformer, 'get_support'):
            return self.feature_names[self.transformer.get_support()]
        return self.feature_names

    def fit(self, data_frame=None, y=None):
        """
        Fit the specified transformer and set column names attribute
        """
        # check if input is valid
        self._validate_input(data_frame)
        # feature names will be used for transform output
        self.feature_names = data_frame.columns
        # fit valid sklearn transformer
        self.transformer.fit(data_frame)
        return self

    def _transform(self, data_frame=None):
        # check if input is valid
        self._validate_input(data_frame)
        # feature names should be the same
        assert all(data_frame.columns == self.feature_names), "column names of the fitted data had different features"
        feature_names = self._get_feature_names()
        data_frame_transformed = self.transformer.transform(data_frame) 
        # check sparsity: output need to be dense array not sparse
        if self.is_sparse:
            data_frame_transformed = data_frame_transformed.toarray()
        return data_frame_transformed, feature_names

    def transform(self, data_frame=None, y=None):
        """
        Apply transformer and cast output as a Pandas DataFrame
        """
        data_frame_transformed, feature_names = self._transform(data_frame)
        if any(feature_names):
            return pandas.DataFrame(data_frame_transformed, columns=feature_names) 
        else:
            return data_frame_transformed


class PandasPipelineWrapper(sklearn.base.TransformerMixin):
    """
    Wrap sklearn Pipeline steps with PandasTransformerWrapper
    """
    def __init__(self, steps):
        self.steps = steps

    def fit(self, data_frame=None, y=None):
        """
        Wrap pipeline transformer steps with PandasTransformerWrapper and fit transformer 
        """
        wrapped_steps = []
        for name, transformer_ in self.steps:
            wrapped_steps.append(
                (name, PandasTransformerWrapper(transformer_))
                )
        self.pipeline = sklearn.pipeline.Pipeline(wrapped_steps)
        self.pipeline.fit(data_frame)
        return self

    def transform(self, data_frame=None, y=None):
        return self.pipeline.transform(data_frame)


class FeatureUnionWrapper(sklearn.base.TransformerMixin):
    """
    Wrap FeatureUnion to persist feature names through pipeline
    
    """
    def __init__(self, steps):
        self.steps = steps

    def fit(self, data_frame=None, y=None):
        wrapped_steps = []
        for name, transformer_ in self.steps:
            wrapped_steps.append(
                (name, PandasTransformerWrapper(transformer_))
                )
        self.union = sklearn.pipeline.FeatureUnion(wrapped_steps)
        self.union.fit(data_frame)   
        return self

    def _transform(self, data_frame=None):
        # check if input is valid
        self._validate_input(data_frame)
        # feature names should be the same
        assert all(data_frame.columns == self.feature_names), "column names of the fitted data had different features"
        feature_names = self._get_feature_names()
        data_frame_transformed = self.transformer.transform(data_frame) 
        # check sparsity: output need to be dense array not sparse
        if self.is_sparse:
            data_frame_transformed = data_frame_transformed.toarray()
        return data_frame_transformed, feature_names

    def transform(self, data_frame=None):
        return self.union.transform(data_frame)

"""
from sklearn_pandas_wrapper import PandasTransformerWrapper, PandasPipelineWrapper
import numpy
import pandas
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn_pandas_wrapper import PandasTransformerWrapper
   
# wrap sklearn transformer
imputer = PandasTransformerWrapper(SimpleImputer, strategy='median')
r = numpy.random.rand(5,3)
r[0, 0] = numpy.nan
df = pandas.DataFrame(r, columns=['a', 'b', 'c'])
# print(df)
# imputer.fit(df)
# print(imputer.transform(df))
# X = pandas.DataFrame([['Male', 1], ['Female', 3], ['Female', 2]], columns=['gender', 'group'])
X = pandas.DataFrame([[0.5, 1], [0.9, 3], [0.6, 2]], columns=['x1', 'x2'])
# enc = Pipeline(steps=[('a', PandasTransformerWrapper(OneHotEncoder())),
#                       ('b', PandasTransformerWrapper(PolynomialFeatures(2))),
#                      ])
steps=[('a', PandasTransformerWrapper(OneHotEncoder())),
       ('b', PandasTransformerWrapper(PolynomialFeatures(2)))
       ]
enc = FeatureUnion(steps)
enc.fit(X)
print(enc.transform(X))
"""
