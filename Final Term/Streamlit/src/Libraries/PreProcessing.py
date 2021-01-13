from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

'''
This class contains all the methods that are used to pre process the dataset and to clean the data.
'''
class PreProcessing:
    cols_to_drop = [
        'health_insurance',
        'employment_industry',
        'employment_occupation',
        'h1n1_concern',
        'rent_or_own',
        'behavioral_large_gatherings',
        'marital_status',
        'education',
        'respondent_id'
        ]
    
    drop_seasonal_end = [
        'chronic_med_condition',
        'sex',
        'employment_status',
        'hhs_geo_region',
        'census_msa',
        'FamilySize',
        'behavior'
    ]
    
    drop_h1n1_end = [
        'chronic_med_condition',
        'child_under_6_months',
        'sex',
        'employment_status',
        'census_msa'
    ]
    
    drop_after_merge = [
        'behavioral_antiviral_meds',
        'behavioral_avoidance',
        'behavioral_face_mask',
        'behavioral_wash_hands',
        'behavioral_outside_home',
        'behavioral_touch_face',
        'household_adults',
        'household_children'
    ]
    
    drop_seasonal = [
       'h1n1_knowledge',
        'doctor_recc_h1n1',
        'child_under_6_months',
        'opinion_h1n1_vacc_effective',
        'opinion_h1n1_risk',
        'opinion_h1n1_sick_from_vacc'
    ]
    
    h1n1_drop = [
        'doctor_recc_seasonal',
        'opinion_seas_vacc_effective',
        'opinion_seas_risk',
        'opinion_seas_sick_from_vacc'
    ]
   
    mode = {'h1n1_concern': 2,
        'h1n1_knowledge': 1,
        'behavioral_antiviral_meds': 0,
        'behavioral_avoidance': 1,
        'behavioral_face_mask': 0,
        'behavioral_wash_hands': 1,
        'behavioral_large_gatherings': 0,
        'behavioral_outside_home': 0,
        'behavioral_touch_face': 1,
        'doctor_recc_h1n1': 0,
        'doctor_recc_seasonal': 0,
        'chronic_med_condition': 0,
        'child_under_6_months': 0,
        'health_worker': 0,
        'opinion_h1n1_vacc_effective': 4,
        'opinion_h1n1_risk': 2,
        'opinion_h1n1_sick_from_vacc': 2,
        'opinion_seas_vacc_effective': 4,
        'opinion_seas_risk': 2,
        'opinion_seas_sick_from_vacc': 1,
        'education': 'College Graduate',
        'marital_status': 'Married',
        'household_adults': 1,
        'household_children': 0,
        'employment_status': 'Employed',
        'income_poverty': '<= $75,000, Above Poverty',
        'rent_or_own': 'Own'}
    
    '''
        This method is used to encode the values of our features
    '''
    def encodeValues(df, feature):
        labelencoder = LabelEncoder()
        df[feature] = labelencoder.fit_transform(df[feature])
        
    '''
        This method is used to encode the ordinal features
    '''   
    def encodeOrdinal(df, feature, mapping):
        df[feature] = df[feature].apply(lambda x: mapping[x])
        
    '''
        This method is used to clean the dataset.
        To clean the dataset we delete some columns, we encode the values of some columns and we add some features.
    '''
    def TestSetCleaning(to_be_cleaned):
        test_dataset = to_be_cleaned.copy()
        labels_seasonal = test_dataset.pop('seasonal_vaccine')
        labels_h1n1 = test_dataset.pop('h1n1_vaccine')
        test_dataset.drop(PreProcessing.cols_to_drop, axis=1, inplace = True)
        
        for col in test_dataset.columns:
            if(col in PreProcessing.mode):
                test_dataset[col].fillna(PreProcessing.mode[col], inplace=True)
        for features in test_dataset.columns:
            if(test_dataset[features].dtypes == 'float64'):
                test_dataset[features] = test_dataset[features].astype(int)
                  
        PreProcessing.encodeValues(test_dataset, 'sex')
        PreProcessing.encodeValues(test_dataset, 'employment_status')
        PreProcessing.encodeValues(test_dataset, 'hhs_geo_region')
        PreProcessing.encodeValues(test_dataset, 'census_msa')
        PreProcessing.encodeOrdinal(test_dataset, 'age_group', {"18 - 34 Years": 0, "35 - 44 Years": 1, "45 - 54 Years": 2, "55 - 64 Years": 3, "65+ Years": 4})
        PreProcessing.encodeOrdinal(test_dataset, 'income_poverty', {"Below Poverty": 0, "<= $75,000, Above Poverty": 1, "> $75,000": 2})
        PreProcessing.encodeOrdinal(test_dataset, 'race', {"Black": 0, "Hispanic": 1, "Other or Multiple": 2, "White": 3})
        
        test_dataset['FamilySize'] = test_dataset['household_adults'].astype(int) + test_dataset['household_children'].astype(int) + 1
        test_dataset['behavior'] = test_dataset['behavioral_antiviral_meds'] + test_dataset['behavioral_avoidance'] + test_dataset['behavioral_face_mask'] + test_dataset['behavioral_wash_hands'] + test_dataset['behavioral_outside_home'] + test_dataset['behavioral_touch_face']
        test_dataset.drop(PreProcessing.drop_after_merge, axis=1, inplace = True)
        seasonalFlu = test_dataset.drop(PreProcessing.drop_seasonal, axis=1)
        h1n1 = test_dataset.drop(PreProcessing.h1n1_drop, axis=1)

        numeric_preprocessing_steps = Pipeline([
            ('standard_scaler', StandardScaler()),
            ('simple_imputer', SimpleImputer(strategy='median'))
        ])

        numeric_cols = seasonalFlu.columns[seasonalFlu.dtypes != "object"].values
        
        preprocessor = ColumnTransformer(
            transformers = [
                ("numeric", numeric_preprocessing_steps, numeric_cols)
            ],
            remainder = "drop"
        )
        
        seasonalFlu.drop(PreProcessing.drop_seasonal_end, axis=1, inplace = True)

        numeric_cols = h1n1.columns[h1n1.dtypes != "object"].values
        
        preprocessor = ColumnTransformer(
            transformers = [
                ("numeric", numeric_preprocessing_steps, numeric_cols)
            ],
            remainder = "drop"
        )
        h1n1.drop(PreProcessing.drop_h1n1_end, axis=1, inplace = True)
        
        for features in seasonalFlu.columns:
            if(seasonalFlu[features].dtypes == 'object'):
                seasonalFlu[features] = seasonalFlu[features].astype(int)
        for features in h1n1.columns:
            if(h1n1[features].dtypes == 'object'):
                h1n1[features] = h1n1[features].astype(int)

        return seasonalFlu, h1n1, labels_seasonal, labels_h1n1  

    '''
        This method is used to clean the dataset without encoding the categorical features
    '''
    def TestSetCleaningNoCod(df):

        test_dataset = df.copy()
        labels_seasonal = test_dataset.pop('seasonal_vaccine')
        labels_h1n1 = test_dataset.pop('h1n1_vaccine')
        test_dataset.drop(PreProcessing.cols_to_drop, axis=1, inplace = True)
        
        for col in test_dataset.columns:
            if(col in PreProcessing.mode):
                test_dataset[col].fillna(PreProcessing.mode[col], inplace=True)
        for features in test_dataset.columns:
            if(test_dataset[features].dtypes == 'float64'):
                test_dataset[features] = test_dataset[features].astype(int)
                
        PreProcessing.encodeValues(test_dataset, 'sex')
        PreProcessing.encodeValues(test_dataset, 'employment_status')

        PreProcessing.encodeValues(test_dataset, 'census_msa')
      
        test_dataset['income_poverty'] = test_dataset['income_poverty'].replace(',','', regex=True, inplace=True)
        
        
        test_dataset['FamilySize'] = test_dataset['household_adults'].astype(int) + test_dataset['household_children'].astype(int) + 1
        test_dataset['behavior'] = test_dataset['behavioral_antiviral_meds'] + test_dataset['behavioral_avoidance'] + test_dataset['behavioral_face_mask'] + test_dataset['behavioral_wash_hands'] + test_dataset['behavioral_outside_home'] + test_dataset['behavioral_touch_face']
        test_dataset.drop(PreProcessing.drop_after_merge, axis=1, inplace = True)
        seasonalFlu = test_dataset.drop(PreProcessing.drop_seasonal, axis=1)
        h1n1 = test_dataset.drop(PreProcessing.h1n1_drop, axis=1)

        numeric_preprocessing_steps = Pipeline([
            ('standard_scaler', StandardScaler()),
            ('simple_imputer', SimpleImputer(strategy='median'))
        ])

        numeric_cols = seasonalFlu.columns[seasonalFlu.dtypes != "object"].values
        
        preprocessor = ColumnTransformer(
            transformers = [
                ("numeric", numeric_preprocessing_steps, numeric_cols)
            ],
            remainder = "drop"
        )
        
        seasonalFlu.drop(PreProcessing.drop_seasonal_end, axis=1, inplace = True)

        numeric_cols = h1n1.columns[h1n1.dtypes != "object"].values
        
        preprocessor = ColumnTransformer(
            transformers = [
                ("numeric", numeric_preprocessing_steps, numeric_cols)
            ],
            remainder = "drop"
        )
        h1n1.drop(PreProcessing.drop_h1n1_end, axis=1, inplace = True)
        
        seasonalFlu = seasonalFlu.loc[:, ~seasonalFlu.columns.str.contains('^Unnamed')]
        h1n1 = h1n1.loc[:, ~h1n1.columns.str.contains('^Unnamed')]

        return seasonalFlu, h1n1, labels_seasonal, labels_h1n1