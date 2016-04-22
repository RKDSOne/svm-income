import numpy as np
import cPickle
from os.path import isfile
from sklearn import preprocessing as pre
from sklearn import svm
from sklearn import metrics
import sys

def preprocess(file_n):
    '''Formats file into np matrix for training'''
    with open(file_n) as f:
        x_data = []
        y_data = []
        
        for line in f:
            tokens = line.split(', ')
            if line != '\n':
                x_data.append(vectorize(tokens))
                y_data.append(int(tokens[-1] == '<=50K\n'))
                
        #normalize values
        x_data = pre.scale(x_data, axis=0)
                
        #divide the data into training and validation    
        div_point = (len(x_data) * 15) / 16        
        x_train = np.array(x_data[:div_point])
        y_train = np.array(y_data[:div_point])
        x_val = np.array(x_data[div_point:])
        y_val = np.array(y_data[div_point:])
        
    return x_train, y_train, x_val, y_val

def preprocess_test(file_n):
    '''Formtas file into np matrix for testing'''
    with open(file_n) as f:
        x_data = []
        
        for line in f:
            tokens = line.split(', ')
            if line != '\n':
                x_data.append(vectorize(tokens))
                
        #normalize values
        x_data = pre.scale(x_data, axis=0)
                     
        x_test = np.array(x_data)
        
    return x_test

def vectorize(array):
    
    #Continuous values
    vector = [
    float(array[0]), #age
    float(array[2]), #fnlwgt
    float(array[4]), #eucation-num
    float(array[10]), #capital gain
    float(array[11]), #capital-loss
    float(array[12]) #hours-per-week
    ]
    
    #categories to be exapnded into boolean vector components
    bool_cats = [ ('workclass', 1),
                  ('marital-status', 5),
                  ('occupation', 6),
                  ('relationship', 7),
                  ('race', 8),
                  ('sex', 9),
                  ('native-country', 10) ]
                                 
    #Append boolean vectors
    if isfile('50knames.txt'):
        #check that name data exists
        bool_vectorize = bool_vectorize_factory('50knames.txt')
        for x, i in bool_cats:
            vector += bool_vectorize(x, array[i])
    else:
        print "50knames.txt not found in current directory"
        sys.exit(1)
        
            
    return vector
    
def bool_vectorize_factory(cat_file):
    '''Returns func that will vectorize a category'''
    with open(cat_file) as file_c:
        categories = {}
        for line in file_c:
            split_line = line.split(' ')
            for i, tokens in enumerate(split_line):
                #removes last character which is usually junk such as . or ,
                split_line[i] = tokens[:-1]
            categories[split_line[0]] = split_line[1:]
            
    def payload(category, token):
        return [float(token == word) for word in categories[category]]
            
    return payload
    
def str_adder_factory(y_pred):
    def inner(i):
        if i < len(y_pred):
            if y_pred[i]:
                return ', <=50K'
            else:              
                return ', >50K'
        return ''
    return inner
    
def main(): 
    if not isfile('model.pkl'):
        if isfile('50kadults.data.txt'):
            x_train, y_train, x_val, y_val = preprocess('50kadults.data.txt')
            clf = svm.SVC(kernel='linear')
            clf.fit(x_train, y_train)
            y_pred = clf.predict(x_val)
            print "Score: " + str(metrics.accuracy_score(y_val, y_pred))    

            with open('model.pkl', 'wb') as pk_file:
                cPickle.dump(clf, pk_file)
        else:
            print "50kadulsts.data.txt not found in current directory."
            sys.exit(1)

    else:
        with open('model.pkl', 'rb') as pk_file:
            clf = cPickle.load(pk_file)
                       
                  
    if len(sys.argv) == 2:
        name = sys.argv[1]
        x_test = preprocess_test(name)
        y_pred = clf.predict(x_test)
        str_adder = str_adder_factory(y_pred)
        with open(name, 'r') as f:    
            file_lines = [''.join([x.strip(), str_adder(i), '\n']) for i, x in enumerate(f.readlines())]
            
        with open('classified_' + name, 'w') as f:
            f.writelines(file_lines)
    else:
        print "No test file arg"
        sys.exit(1)

    
    
if __name__ == '__main__':
    main()