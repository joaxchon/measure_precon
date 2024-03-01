import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt

import ot
import random

test_number = 444
from scipy.ndimage import gaussian_filter


mnist = tf.keras.datasets.mnist
#mnist is a class module type

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# after load x_train is a numpy array
x_train, x_test = x_train / 255.0, x_test / 255.0

y_train = np.array(y_train)
x_train = np.array(x_train)
y_test = np.array(y_test)
x_test = np.array(x_test)
idx = [i for i in range(len(y_train)) if y_train[i] == 1 or y_train[i] == 9]
y_train_mod = y_train[idx]
x_train_mod = x_train[idx]

idx_test = [i for i in range(len(y_test)) if y_test[i] == 1 or y_test[i] == 9]
y_test_mod = y_test[idx_test]
x_test_mod = x_test[idx_test]


idx_target_train = [i for i in range(len(y_train)) if y_train[i] == 7 or y_train[i] == 6]
x_train_target = x_train[idx_target_train]
y_train_target = y_train[idx_target_train]

idx_target_test = [i for i in range(len(y_test)) if y_test[i] == 7 or y_test[i] == 6]
x_test_target = x_test[idx_target_test]
y_test_target = y_test[idx_target_test]


fig = plt.figure
plt.imshow(x_test_target[test_number], cmap='Blues')
plt.show()



#### Here at the array level I can pre-condition x_train

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
    ############### adds the dimension axis.
  tf.keras.layers.Dense(128, activation='relu'),
    ########### dense implements (weight*input + bias)
  tf.keras.layers.Dropout(0.2),
    ########## droput sets 0s to avoid overfitting
  tf.keras.layers.Dense(10)
    #########3 second dense layer
])

#predictions = model(x_train[:1]).numpy()
########################### prediction modifications
predictions = model(x_train_mod[:1]).numpy()
#print(predictions)

tf.nn.softmax(predictions).numpy()

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
#loss_fn = tf.keras.losses.MeanAbsoluteError()
#loss_fn = tf.keras.losses.MeanSquaredError()

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

#model.fit(x_train, y_train, epochs=5)
model.fit(x_train_mod, y_train_mod, epochs=5)

#a = np.array(model.layers[1].get_weights())
#model.layers[1].set_weights(a + 1000)
#b = np.array(model.layers[1].get_weights())
#print(b)


#model.evaluate(x_test,  y_test, verbose=2)
print('Moment of evaluation of tests ...................................................................')
model.evaluate(x_test_mod,y_test_mod,verbose=2)
print('Evaluation is over -------------------------------------------------------------------------------')

probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])
#print(x_test[1:5:2].shape)
#print(probability_model(x_test[1:5]))

#weights = model.layers.get_weights()
#rint(type(weights))
#print(len(model.layers))
model.summary()
#weights_1 = model.layers[1].get_weights()
#print(len(weights_1))
#print(weights_1[0:])
################# REWRITE LAYER WEIGHTS at every layer?

#########################################################################
#########################################################################
##################### Step 1: Computation of Mij for y = A

idx_test_A = [i for i in range(len(y_test_mod)) if y_test_mod[i] == 1]
y_test_mod_A = y_test_mod[idx_test_A]
x_test_mod_A = x_test_mod[idx_test_A]

idx_target_test_A = [i for i in range(len(y_test_target)) if y_test_target[i] == 7]
x_test_target_A = x_test_target[idx_target_test_A]
y_test_target_A = y_test_target[idx_target_test_A]


size_to_transport_A = min(x_test_target_A.shape[0], x_test_target_A.shape[0])
x_test_mod_A_transport = x_test_mod_A[:size_to_transport_A,:,:]
x_test_target_A_transport = x_test_target_A[:size_to_transport_A,:,:]
x_test_mod_A_transport = x_test_mod_A_transport.reshape(size_to_transport_A,784)
x_test_target_A_transport = x_test_target_A_transport.reshape(size_to_transport_A,784)
ot_emd_A = ot.da.EMDTransport()
ot_emd_A.fit(Xs=x_test_mod_A_transport,Xt=x_test_target_A_transport)
Xs_mapped_A = ot_emd_A.transform(Xs=x_test_mod_A_transport)

domain_elements_A = ot_emd_A.inverse_transform(Xt =x_test_target_A_transport)

##################### Step 1: Computation of Mij for y = b

idx_test_B = [i for i in range(len(y_test_mod)) if y_test_mod[i] == 9]
y_test_mod_B = y_test_mod[idx_test_B]
x_test_mod_B = x_test_mod[idx_test_B]

idx_target_test_B = [i for i in range(len(y_test_target)) if y_test_target[i] == 6]
x_test_target_B = x_test_target[idx_target_test_B]
y_test_target_B = y_test_target[idx_target_test_B]


size_to_transport_B = min(x_test_target_B.shape[0], x_test_target_B.shape[0])
x_test_mod_B_transport = x_test_mod_B[:size_to_transport_B,:,:]
x_test_target_B_transport = x_test_target_B[:size_to_transport_B,:,:]
x_test_mod_B_transport = x_test_mod_B_transport.reshape(size_to_transport_B,784)
x_test_target_B_transport = x_test_target_B_transport.reshape(size_to_transport_B,784)
ot_emd_B = ot.da.EMDTransport()
ot_emd_B.fit(Xs=x_test_mod_B_transport,Xt=x_test_target_B_transport)
Xs_mapped_B = ot_emd_B.transform(Xs=x_test_mod_B_transport)

domain_elements_B= ot_emd_B.inverse_transform(Xt =x_test_target_B_transport)



#new_value_to_predict = 1/2*(x_test_target_A_transport[2,:] + x_test_target_B_transport[46,:] )
#new_value_to_predict =  x_test_target_B_transport[test_number,:]
new_value_to_predict =  x_test_target_B_transport[test_number,:]

print('Aquiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii' + str(new_value_to_predict.shape))
fig = plt.figure
plt.imshow(new_value_to_predict.reshape(28,28), cmap='Reds')
plt.show()


dis_A=[]
dist_A = [0 for i in range(x_test_target_A_transport.shape[0])]
################### compute distances to 7 and 9
for i in range(x_test_target_A_transport.shape[0]):
    dist_A[i] = np.linalg.norm(x_test_target_A_transport[i,:] - new_value_to_predict)

#print(dist_A)
minimum_to_A = dist_A.index(min(dist_A))
print(minimum_to_A)
closest_A = x_test_target_A_transport[minimum_to_A,:]


dis_B=[]
dist_B = [0 for i in range(x_test_target_B_transport.shape[0])]
################### compute distances to 7 and 9
for i in range(x_test_target_B_transport.shape[0]):
    dist_B[i] = np.linalg.norm(x_test_target_B_transport[i,:] - np.flip(new_value_to_predict),1)

minimum_to_B = dist_B.index(min(dist_B))

print(minimum_to_B)

convex_value = (dist_A[minimum_to_A]) / (dist_A[minimum_to_A] + dist_B[minimum_to_B])


predicted_value = model.predict(domain_elements_A[range(minimum_to_A,minimum_to_A+1),:].reshape(1,28,28)) * convex_value + model.predict(np.flip(domain_elements_B[range(minimum_to_B,minimum_to_B+1),:].reshape(1,28,28),1))* (1-convex_value)
print( model.predict(domain_elements_A[range(minimum_to_A,minimum_to_A+1),:].reshape(1,28,28)) )
print(convex_value)
print(model.predict(np.flip(domain_elements_B[range(minimum_to_B,minimum_to_B+1),:].reshape(1,28,28),1)))
print(predicted_value)







