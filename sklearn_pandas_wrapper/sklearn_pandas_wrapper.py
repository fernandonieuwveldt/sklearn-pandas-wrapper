import sklearn
import pandas
 

# Todo: Check for special cases where number of features change after transform
#       For example OneHotEncoder
# Todo: Add functionality for pipelines
class PandasTransformerWrapper(sklearn.base.TransformerMixin):
    """
    Wrap sklearn Transformer return type from numpy to pandas with column names
    if Pandas DataFrame given as input
    """
    def __init__(self, transformer=None, **kwargs):
        self._validate_transformer(transformer)
        if callable(transformer):
            self.transformer = transformer(**kwargs)
        else:
            self.transformer = transformer

    def _validate_transformer(self, transformer=None):
        """
        Check if transformer is a valid sklearn transformer
        """
        is_transformer = all(map(lambda method: hasattr(transformer, method), ['fit', 'transform', 'fit_transform']))
        if not is_transformer:
            raise ValueError('transformer does not contain all of fit, transform and fit_transform methods')

    def _validate_input(self, data_frame=None):
        """
        Input should be a data frame. Check type of input and raise error if not a pandas data frame
        """
        if isinstance(data_frame, pandas.core.frame.DataFrame):
            return True
        else:
            raise ValueError('Input should be a pandas data frame ')

    def fit(self, data_frame=None):
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

    def transform(self, data_frame=None):
        """
        Apply transformer and cast output as a Pandas DataFrame
        """
        # check if input is valid
        self._validate_input(data_frame)
        # feature names should be the same
        assert all(data_frame.columns == self.feature_names), "column names of the fitted data had different features"
        return pandas.DataFrame(self.transformer.transform(data_frame),
                                columns=self.feature_names) 
