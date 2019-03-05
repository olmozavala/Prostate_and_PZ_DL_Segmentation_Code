from os.path import join
from os import walk, listdir
from pandas import Series

def saveDictionaryToFile(my_dict, file_name):
    '''This function will write the values of a dictionary into a csv, BUT it will also
    append the mean value as the last row'''
    data= Series(my_dict, index=my_dict.keys())
    mean_value = data.mean()
    data['AVG'] = mean_value
    data.sort_index(axis=0,inplace=True)
    data.to_csv(file_name)

def saveMultipleDictionaryToFile(all_dicts, file_name, names):
    '''This function will write all the values of a dictionary into a csv, BUT it will also
    append the mean value as the last row'''
    for ii, my_dict in enumerate(all_dicts):
        if ii == 0:
            data= Series(my_dict, index=my_dict.keys())

    mean_value = data.mean()
    data['AVG'] = mean_value
    data.sort_index(axis=0,inplace=True)
    data.to_csv(file_name)
