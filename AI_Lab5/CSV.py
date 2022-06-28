import numpy as np
import pandas
from pandas import DataFrame
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


class CSV:
    def __init__(self,path):
        self.path=path
        self.data,self.laber=self.read_data()

    def read_data(self):
        file = pandas.read_csv( self.path, header=None)
        file = file.values
        data, label = file[:, 1:-1], file[:, -1]

        label = LabelEncoder().fit_transform(label)
        print(f'data: {data}')
        print(f'label: {label}')
        array=[0,0,0,0,0,0]
        for i in label:
            #print (i)
            array[i]+=1
        print("Distribution is : ",array)
        #print(len(label))
        return data, label

    def update_max(self,max,j,index,max_index):
        if j > max:
            return j,index
        return max,max_index

    def calc_softmax_pred(self,predictions):
        predicted_labels = []
        counter=0
        for x in predictions:
            index = 0
            max_index = 0
            max = -1
            max_array = np.exp(x) / np.sum(np.exp(x), axis=0)

            print(f'{counter}:   {max_array}')
            for j in max_array:
                max,max_index=self.update_max(max,j,index,max_index)
                index += 1

            predicted_labels.append(max_index)
            counter += 1
        print(f'softmax result is::   {predicted_labels}')
        return predicted_labels

    def calc_micro(self,predicted_labels,test_vec):
        answers = [0, 0, 0, 0, 0, 0]
        results = [0, 0, 0, 0, 0, 0]
        TP = 0
        for predection, test in zip(predicted_labels, test_vec):
            answers[predection] += 1
            if predection == test:
                results[predection] += 1
                TP += 1
        print(f'Micro Result is: {TP / 43}')
        return answers,results

    def calc_macro(self,number_of_answer,number_of_correct):
        final_result = 0
        for answer, result in zip(number_of_answer, number_of_correct):
            final_result += (result/answer)

        print(f'Macro result is : {final_result / 6}')

    def calc_acc(self,train,train_vec,test,test_vec):
        mlp = MLPClassifier(random_state=1, max_iter=8000000).fit(train, train_vec)
        #print(mlp)
        predict = mlp.predict_proba(test)
        #print(predict)
        predicted_labels=self.calc_softmax_pred(predict)
        number_of_answer,number_of_correct =self.calc_micro(predicted_labels,test_vec)
        self.calc_macro(number_of_answer,number_of_correct)


