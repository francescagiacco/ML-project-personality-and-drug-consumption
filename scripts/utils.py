import pandas as pd
import  sklearn

def create_dictionary(column_data, d):
    '''input : data = column of pandas dataframe
                d = dictionary

        output : dictionary, keys: keys of d (d[i])
                            value : list of 0 and 1 '''

    output = {}
    for race in d:
        #print(d[race])
        output[race] =  [1  if i == d[race] else 0 for i in column_data]

    return output



def append_in_dataframe(data, dictionary):
    ''' input: data = pandas dataframe
                dictionary = dictionary, keys: race-
                                        value : binary list (0,1)
        output: pandas dataframe + (keys of d)_columns with the list as values'''
    new_data = data.copy()
    for new_column in dictionary:
        #print(new_column, dictionary[new_column])
        new_data[new_column] = dictionary[new_column]
        #print(new_data[new_column])
    return new_data

if __name__ == '__main__':
    data = pd.read_csv('..\drug_consumption.csv')
    d = {'asian' : -0.50212, 'black_asian' : 1.90725, 'white_asian' : 0.126001,
     'white_black' : -0.22166, 'other' : 0.11440, 'white' : -0.31685,
     'other2': -1.10702 }


    dictionary = create_dictionary(data['etnicity'], d)
    final_df = append_in_dataframe(data, dictionary)


    #remove the 'etnicity' column
    final_df = final_df.drop('etnicity', axis = 1).head()
    final_df.head()
    print(final_df.head())
