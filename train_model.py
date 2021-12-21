# USAGE
# python train_model.py --embeddings output/embeddings.pickle --recognizer output/recognizer.pickle --le output/le.pickle
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# import the necessary packages
from sklearn.preprocessing import LabelEncoder
# from sklearn.svm import SVC
# import argparse
import pickle
# from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
# from sklearn import svm
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_absolute_error
# from sklearn.metrics import classification_report, accuracy_score
# import tensorflow as tf
# import keras
from tensorflow.keras.callbacks import ModelCheckpoint
# import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error




# train the model used to accept the 128-d embeddings of the face and
# then produce the actual face recognition
# recognizer = svm.SVC(C=1.0, kernel="linear", probability=True)


for i in range(0, 3):
     if(i>0):
         print("[INFO] training model...")
         # load the face embeddings
         print("[INFO] loading face embeddings...")
         data = pickle.loads(open("output/embeddings.pickle", "rb").read())
         # # encode the labels
         print("[INFO] encoding labels...")
         le = LabelEncoder()
         labels = le.fit_transform(data["names"])

         x_train, x_test, y_train, y_test = train_test_split(data["embeddings"], labels, random_state=0, test_size=0.25)

         recognizer = RandomForestClassifier(warm_start=True)

         recognizer.fit(data["embeddings"], labels)

         pred_y = recognizer.predict(data["embeddings"])
         mae = mean_absolute_error(pred_y, labels)
         print("mae      :", mae)
         print('pred_y :', pred_y)


         recognizer.n_estimators += 1
         recognizer.fit(data["embeddings"], labels)

         pred_y = recognizer.predict(data["embeddings"])
         mae = mean_absolute_error(pred_y, labels)
         print("mae      :", mae)
         print('pred_y :', pred_y)



         # write the actual face recognition model to disk
         f = open("output/recognizer.pickle", "wb")
         f.write(pickle.dumps(recognizer))
         f.close()

         # write the label encoder to disk
         f = open("output/le.pickle", "wb")
         f.write(pickle.dumps(le))
         f.close()

# checkpointer = ModelCheckpoint(filepath="openface_nn4.small2.v1.t7", save_best_only = True )
# for i in range(0, 3):
#     if(i>0):
#         print("[INFO] training model...")
#
#         # load the face embeddings
#         print("[INFO] loading face embeddings...")
#         data = pickle.loads(open("output/embeddings.pickle", "rb").read())
#
#         # # encode the labels
#         print("[INFO] encoding labels...")
#         le = LabelEncoder()
#         labels = le.fit_transform(data["names"])
#
#         recognizer = RandomForestClassifier(n_estimators=1, warm_start=True)
#         recognizer.fit(data["embeddings"], labels)
#         recognizer.n_estimators += 1
#         recognizer.fit(data["embeddings"], labels)
#         # joblib.dump('recognizer'+str(i)+'.pickle')
#
#         # write the actual face recognition model to disk
#         f = open("output/recognizer.pickle", "wb")
#         f.write(pickle.dumps(recognizer))
#         f.close()
#
#         # write the label encoder to disk
#         f = open("output/le.pickle", "wb")
#         f.write(pickle.dumps(le))
#         f.close()

# cross_val_score(recognizer, data["embeddings"], labels, scoring='recall_macro')
# print(cross_val_score)
# View accuracy score


# Instantiate and fit the RandomForestClassifier
# recognizer = RandomForestClassifier()
# recognizer.fit(x_train, y_train)

# Make predictions for the test set
y_pred_test = recognizer.predict(x_test)

# print(accuracy_score(y_test, y_pred_test))