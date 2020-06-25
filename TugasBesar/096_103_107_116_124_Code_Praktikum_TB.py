# -*- coding: utf-8 -*-
"""
Created on Thu May 21 20:27:07 2020

@author: User
"""
import pandas as pd
import numpy as np
kolom = ['class','cap_shape','cap_surface','cap_color','bruises','odor','gill_attachment','gill_spacing',
         'gill_size','gill_color','stalk_shape','stalk_root','stalk_surface_above_ring',
         'stalk_surface_below_ring','stalk_color_above_ring','stalk_color_below_ring',
         'veil_type','veil_color','ring_number','ring_type','spore_print_color','population','habitat']
#load dataset
jamur= pd.read_csv("mushrooms.csv",names=kolom)

#pisah data, dari fitur dan kelasnya
fitur = ['cap_shape','cap_surface','cap_color','bruises','odor','gill_attachment','gill_spacing',
         'gill_size','gill_color','stalk_shape','stalk_root','stalk_surface_above_ring',
         'stalk_surface_below_ring','stalk_color_above_ring','stalk_color_below_ring',
         'veil_type','veil_color','ring_number','ring_type','spore_print_color','population','habitat']

X = jamur[fitur] #features
Class = jamur['class'] #target values

#mengisi data '?' menjadi kosong
arrayData = jamur.values
j = arrayData[:,0:23] 
j[j=='?']=''

from sklearn import preprocessing
#membuat label encoder: mengubah variable string ke angka (sesuai urutan huruf)
le = preprocessing.LabelEncoder()
# bell ;0,conical ;1, flat ;2, knobbed ;3, sunken ;4,convex:5
X.cap_shape=le.fit_transform(X.cap_shape)
# fibrous=0,grooves=1, smooth=2, scaly=3
X.cap_surface=le.fit_transform(X.cap_surface)
#buff=0,cinnamon=1,red=2,green=3,brown=4,pink=5,green=6,purple=7,white=8,yellow=9
X.cap_color=le.fit_transform(X.cap_color)
#no=0,bruises=1
X.bruises=le.fit_transform(X.bruises)
#almond=0,creosote=1,foul=2,anise=3,musty=4,none=5,pungent=6,spicy=7,fishy=8
X.odor=le.fit_transform(X.odor)
#attached=0,descending=1,free=2,nothced=3
X.gill_attachment=le.fit_transform(X.gill_attachment)
#close=0,distant=1,crowded=2
X.gill_spacing=le.fit_transform(X.gill_spacing)
#board=0,narrow=1
X.gill_size=le.fit_transform(X.gill_size)
#buff=0,red=1,gray=2,chocolate=3,black=4,brown=5,orange=6,pink=7,green=8,purple=9,white=10,yellow=11
X.gill_color=le.fit_transform(X.gill_color)
#enlarging=0,tapering=1
X.stalk_shape=le.fit_transform(X.stalk_shape)
#bulbous=0,club=1,equal=2,rooted=3,rhizomorps=4,missing=
X.stalk_root=le.fit_transform(X.stalk_root)
#fibrous=0,silky=1,smooth=2,scaly=3
X.stalk_surface_above_ring=le.fit_transform(X.stalk_surface_above_ring)
#fibrous=0,silky=1,smooth=2,scaly=3
X.stalk_surface_below_ring=le.fit_transform(X.stalk_surface_below_ring)
#buff=0,cinnamon=1,red=2,gray=3,brown=4,orange=5,pink=6,white=7,yellow=8
X.stalk_color_above_ring=le.fit_transform(X.stalk_color_above_ring)
#buff=0,cinnamon=1,red=2,gray=3,brown=4,orange=5,pink=6,white=7,yellow=8
X.stalk_color_below_ring=le.fit_transform(X.stalk_color_below_ring)
#partial=0,universal=1
X.veil_type=le.fit_transform(X.veil_type)
#brown=0,orange=1,white=2,yellow=3
X.veil_color=le.fit_transform(X.veil_color)
#none=0,one=1,two=2
X.ring_number=le.fit_transform(X.ring_number)
#cobwebby=0,evanescent=1,flaring=2,large=3,none=4,pendant=5,sheating=6,zone=7
X.ring_type=le.fit_transform(X.ring_type)
#buff=0,choclate=1,black=2,brown=3,orange=4,green=5,purple=6,white=7,yellow=8
X.spore_print_color=le.fit_transform(X.spore_print_color)
#abundant=0,clustered=1,numerous=2,scattered=3,several=4,solitary=5
X.population=le.fit_transform(X.population)
#woods=0,grasses=1,leaves=2,meadows=3,paths=4,urban=5,waste=6
X.habitat=le.fit_transform(X.habitat)
#e=0, p=1
Target = le.fit_transform(Class)

X_data = np.array(X)
Y_data = Target

from sklearn.tree import DecisionTreeClassifier #import decision tree classifier
#panggil decision tree 
clf = DecisionTreeClassifier()
#terapkan decision tree pada data x, dan kelas y
clf = clf.fit(X,Class)

#prediksi kelas mushrooms untuk data 1 = convex, smooth, brown, bruises, pungent, free, close, 
#narrow, black, enlarging, equal, smooth, smooth, white, white, partial, white, orange, pendant, 
#black, scattered, urban
y_pred1 = clf.predict([[5,2,4,1,6,2,0,1,4,0,2,2,2,7,7,0,2,1,5,2,3,5]])
print('prediksi untuk Data ke 1 =',y_pred1)

#prediksi kelas mushrooms untuk data 2 = convex, smooth, yellow, bruises, almond, free, close, 
#broad, black, enlarging, club, smooth, smooth, white, white, partial, white, orange, pendant, 
#brown, numerous, grasses
y_pred2 = clf.predict([[5,2,9,1,0,2,0,0,4,0,1,2,2,7,7,0,2,1,5,3,2,1]])
print('prediksi untuk Data ke 2 =',y_pred2)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.20, random_state=0)

from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()

model.fit(X_train, y_train)

predicted = model.predict(X_test)

error=((y_test!=predicted).sum()/len(predicted))*100
print("Error prediksi = %.2f" %error, "%")

akurasi=100-error
print("Akurasi menggunakan klasifikasi Decision Tree = %.2f" %akurasi, "%")

print("Evaluasi Model menggunakan metode Hold Out Estimation : ")
def Conf_matrix(y_actual, y_pred):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    
    for i in range(len(y_pred)):
        if y_actual[i]==y_pred[i]==1:
            TP += 1
        if y_pred[i]==1 and y_actual[i] !=y_pred[i]:
            FP += 1
        if y_actual[i]==y_pred[i]==0:
            TN += 1
        if y_pred[i]==0 and y_actual[i]!= y_pred[i]:
            FN += 1
            
    return (TP, FN, TN, FP)

TP, FN, TN, FP = Conf_matrix(y_test, predicted)

print('akurasi = ', (float)(TP+TN)/(TP+TN+FP+FN))
print('sensitivity = ',(float) (TP+0)/(TP+FN))
print('specificity = ',(float) (TN+0)/(TN+FP))   

from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
from IPython.display import Image
import pydotplus

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True,feature_names=fitur, class_names=['e','p'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('mushrooms.png')
Image(graph.create_png())
