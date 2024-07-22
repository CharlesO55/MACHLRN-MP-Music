import pandas as pd;
import numpy as np;
from sklearn.preprocessing import MinMaxScaler, StandardScaler;


class DataLoader:
    '''
        DataLoader opens csv files to modify and store it to a df dictionary.


        `df` Keys:
            'raw'           - Data after all preprocessing.
            'normalize'     - MinMax Scaling applied on features.
            'standardize'   - Standardization applied on features.

        Preprocessing `dict_df` Keys:
            `raw`             - Original csv as is.
            `imputed`         - Replaced missing (NaN) values.
            `feature_eng`     - Extracted additional info

    '''
    def __init__(self, file_address, perform_Preprocessing, perform_EDA_Features):
        '''
            Using pandas, open the csv file and store to key raw.
        '''
        self.dict_df = dict(raw = pd.read_csv(file_address));
        
        if perform_Preprocessing:
            self.do_Imputation();
            self.do_Feature_Engineering_Track_Names();
            self.do_Feature_Engineering_Artist_Genres();

        if perform_EDA_Features:
            # Get the Preprocessing df
            df = self.dict_df['feature_eng'].copy()

            # Drop the features that are strings/similar between classes
            cols_to_drop = ['Artist Name', 'Track Name', 'time_signature'] + list(df.columns[df.columns.get_loc('Class') + 1:])
            df = df.drop(columns=cols_to_drop);

            # Convert the time (ms) to seconds.
            df['duration_in min/ms'] = df['duration_in min/ms'] / 1000
            df.rename(columns={'duration_in min/ms' : 'duration (seconds)'}, inplace=True)

            # Store the raw
            self.df = dict()
            self.df['raw'] = df

            # Feature scale the rest
            cols_To_Scale = df.columns[:-1]
            
            self.df['normalize'] = self.do_Scaling(MinMaxScaler(), df.copy(), cols_To_Scale);
            self.df['standardize'] = self.do_Scaling(StandardScaler(), df.copy(), cols_To_Scale);

    
    
    def do_Imputation(self):
        # Find the rows with missing values
        mask = self.dict_df['raw'].isna().any(axis=1)
    
        # Generate the median estimates
        self.estimates = self.dict_df['raw'].groupby('Class')[['key', 'instrumentalness', 'Popularity']].median()

        # Create a copy to work on
        self.dict_df['imputed'] = self.dict_df['raw'].copy()
        
        # Replace all NaN values under `key` and `instrumentalness`
        self.dict_df['imputed'] = self.dict_df['imputed'].apply(self.impute_Rows, axis=1)

    def is_NaN(self, x):
        return x != x;
        
    def impute_Rows(self, X, cols_to_Impute = ['key', 'instrumentalness', 'Popularity']):
        for col in cols_to_Impute:
            if self.is_NaN(X[col]):
                X[col] = self.estimates.loc[X['Class'], col]
      
        return X
        

    def do_Feature_Engineering_Track_Names(self, strings_to_Check : dict = None):
        # WORK WITH VIA A COPY
        df = self.dict_df['imputed'].copy()

        # DEFAULT FEATURE ENGINEERING ON TRACK NAME
        if strings_to_Check is None:
            strings_to_Check = { 
                'Is_Collab' :  ['feat', 'Feat', ' ft'],
                'Is_Remix' : ['remix', 'Remix', ' version'] 
            }

        
        # CREATE THE COLS WITH ZERO AS DEFAULT
        df[list(strings_to_Check.keys())] = 0 

        # GO THROUGH EACH COL
        for key in strings_to_Check:
            # GO THROUGH EACH KEYWORD
            for text in strings_to_Check[key]:
                
                # FIND THOSE ROWS WITH THE KEYWORD AND FLAG THEM AS 1
                found_rows = df.query("`Track Name`.str.contains(@text)").index;
                df.loc[found_rows, key] = 1


        # SAVE THE RESULTS TO `dict_df['feature_eng']`
        self.dict_df['feature_eng'] = df


    def do_Feature_Engineering_Artist_Genres(self):
        self.df_artists = self.dict_df['feature_eng'][['Artist Name', 'Class']].groupby('Artist Name')['Class'].unique()

        # COPY FOR MODDING
        df = self.dict_df['feature_eng'].copy()

        # CREATE THE HAS SUNG FLAGS AND INIT AS 0
        has_Sung_Cols = []
        for i in range(0, 11):
            has_Sung_Cols.append('HasSung_' + str(i))
        
        df.loc[:, has_Sung_Cols] = 0

        # MARK THE FLAGS AND UPDATE `dict_df['feature_eng']`
        self.dict_df['feature_eng'] = df.apply(self.flag_Sung_Genres, axis=1)
        

    def flag_Sung_Genres(self, row):
        artist = row['Artist Name']
    
        for genre in self.df_artists[artist]:            
            colIndex = genre - 10
            row['HasSung_' + str(genre)] = 1

        # print(artist)
        return row
    
    def do_Feature_Selection(self, cols_To_Drop = ['Artist Name', 'Track Name']):
        self.dict_df['feature_eng'].drop(columns=cols_To_Drop, inplace=True)



    def do_Scaling(self, scaler, df, cols_To_Scale):
            df[cols_To_Scale] = scaler.fit_transform(df[cols_To_Scale])
            return df;