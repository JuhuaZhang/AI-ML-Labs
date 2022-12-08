# fully connected layer
dnn = Dense(6)(dnn)
dnn = BatchNormalization(axis=-1)(dnn)
dnn = Activation('sigmoid')(dnn)
dnn = Dropout(0.25)(dnn)