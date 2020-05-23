
# coding: utf-8

# Carlos Mougan
import sklearn
import category_encoders
from sklearn.base import BaseEstimator, TransformerMixin

class Encodings(BaseEstimator, TransformerMixin):
    '''
    This class implements fit and transform methods that allows to encode categorical features in different ways.
    
    '''
    
    def __init__(self, encoding_type="TargetEncoder",columns="All",return_categorical=True):
        #cols: list -> a list of columns to encode, if All, all string columns will be encoded.
        
        self._allowed_encodings = ["TargetEncoder","WOEEncoder","CatBoostEncoder","OneHotEncoder"]           
        assert encoding_type in self._allowed_encodings, "the encoding type introduced {} is not valid. Please use one in {}".format(encoding_type, self._allowed_encodings)
        self.encoding_type = encoding_type
        
        self.columns = columns
        self.return_categorical = return_categorical
        
        
    def fit(self,X,y):
        """
        This method learns encodings for categorical variables/values.
        """
        
        #import pdb;pdb.set_trace()
        
        # Obtain a list of categorical variables
        if self.columns == "All":
            self.categorical_cols = X.columns[X.dtypes==object].tolist() +  X.columns[X.dtypes=="category"].tolist()
        else:
            self.categorical_cols = self.columns
        
    
        # Split the data into categorical and numerical
        self.data_encode = X[self.categorical_cols]

        
        # Select the type of encoder
        if self.encoding_type == "TargetEncoder":
            self.enc = category_encoders.target_encoder.TargetEncoder()
            
        if self.encoding_type == "WOEEncoder":
            self.enc = category_encoders.woe.WOEEncoder()
            
        if self.encoding_type == "CatBoostEncoder":
            #This is very similar to leave-one-out encoding, 
            #but calculates the values “on-the-fly”.
            #Consequently, the values naturally vary during the training phase and it is not necessary to add random noise.
            # Needs to be randomly permuted
            # Random permutation
            perm = np.random.permutation(len(X))
            self.data_encode = self.data_encode.iloc[perm].reset_index(drop=True)
            y = y.iloc[perm].reset_index(drop=True)
            self.enc = category_encoders.cat_boost.CatBoostEncoder()
            
        if self.encoding_type == "OneHotEncoder":
            self.enc = category_encoders.one_hot.OneHotEncoder()
            
            # Check if all columns have certain number of elements bf OHE
            self.new_list=[]
            for col in self.data_encode.columns:
                if len(self.data_encode[col].unique())<50:
                    self.new_list.append(col)
                    
            self.data_encode = self.data_encode[self.new_list]
        
        # Fit the encoder
        self.enc.fit(self.data_encode,y)
        return self

    def transform(self, X):
        
        
        if self.columns == "All":
            self.categorical_cols = X.columns[X.dtypes==object].tolist() +  X.columns[X.dtypes=="category"].tolist()
        else:
            self.categorical_cols = self.columns
        
       
    
        # Split the data into categorical and numerical
        
        self.data_encode = X[self.categorical_cols]
        
        # Transform the data
        self.transformed = self.enc.transform(self.data_encode)
        
        # Modify the names of the columns with the proper suffix
        self.new_names = []
        for c in self.transformed.columns:
            self.new_names.append(c+'_'+self.encoding_type)
        self.transformed.columns = self.new_names
         
        if self.return_categorical:
            #print('The encoding {} has made {} columns, the input was {} and the output shape{}'.
             #     format(self.encoding_type,self.transformed.shape, X.shape,self.transformed.join(X).shape))
            #print(self.transformed.join(X).dtypes)

            return self.transformed.join(X)
        else:
            return self.transformed.join(X)._get_numeric_data()

class NaNtreatment(BaseEstimator, TransformerMixin):
    '''
    This class implements a fit and transform methods that enables to implace NaNs in different ways.
    '''
    def __init__(self, treatment="mean"):
        self._allowed_treatments = ["fixed_value", "mean",'median','mode','None']     
        assert treatment in self._allowed_treatments or isinstance(treatment,(int,float)),  "the treatment introduced {} is not valid. Please use one in {}".format(treatment, self._allowed_treatments)
        self.treatment = treatment
    
    def fit(self, X, y):
        """
        Learns statistics to impute nans.
        """
        
        if self.treatment == "mean" or self.treatment==None:
            self.treatment_method = sklearn.impute.SimpleImputer(missing_values=np.nan, strategy='mean')
        elif self.treatment == "median":
            self.treatment_method = sklearn.impute.SimpleImputer(missing_values=np.nan, strategy='median')
        elif self.treatment == "most_frequent":
            self.treatment_method = sklearn.impute.SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        elif isinstance(self.treatment, (int,float)):
            self.treatment_method = sklearn.impute.SimpleImputer(missing_values=np.nan,
                                                                 strategy="constant",fill_value=self.treatment)       
        

        self.treatment_method.fit(X.values)
        return self

    def transform(self, X):
        if self.treatment==None:
            return X
        return self.treatment_method.transform(X)

