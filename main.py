import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt
import random


from scipy.ndimage import gaussian_filter


#mnist is a class module type


# after load x_train is a numpy array


#x_train = gaussian_filter(x_train,sigma = 0.1 )
for sigma_filter_index in range(0,10):
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    x_test = gaussian_filter(x_test,sigma= 1- sigma_filter_index/10)
#x_train = gaussian_filter(x_train, sigma = 0.6)

#for i in range(60000):
#    for j in range(28):
#        for k in range(28):
#            if x_train[i,j,k] < 0.25:
#                x_train[i,j,k] = 0
#            else:
#                pass

            #x_train[i,j,k] = min(x_train[i,j,k],random.uniform(0,1))



    #fig = plt.figure
    #plt.imshow(x_train[58200], cmap='Blues')
    #plt.title(f"Iteration corresponding to sigma =  : {1 -sigma_filter_index/10}.")
    #plt.axis("off")
    #plt.show()

#### Here at the array level I can pre-condition x_train

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10)
    ])

    predictions = model(x_train[:1]).numpy()
#print(predictions)

    tf.nn.softmax(predictions).numpy()

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
#loss_fn = tf.keras.losses.MeanAbsoluteError()
#loss_fn = tf.keras.losses.MeanSquaredError()

    model.compile(optimizer='adam',
                  loss=loss_fn,
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train, epochs=5)
    plt.plot(history.history['accuracy'])
    plt.title(f'model accuracy')

    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    #plt.show()
    model.evaluate(x_test,  y_test, verbose=2)
#print(model.evaluate(x_test,  y_test, verbose=2))

    probability_model = tf.keras.Sequential([
        model,
        tf.keras.layers.Softmax()
    ])
#print(x_test[1:5:2].shape)
#print(probability_model(x_test[1:5]))

    #sample = 1
    #image = x_test[sample]

    #fig = plt.figure()
    #plt.title(f"Iteration corresponding to sigma =  : {1 - sigma_filter_index / 10}.")
    #plt.axis('off')
    #for i in range(6):
      #  plottable_image = np.reshape(x_test[i], (28, 28))
      #  ax = fig.add_subplot(2, 3, i+1)
      #  ax.imshow(plottable_image, cmap='Blues')

       # ax.axis('off')

    #plt.show()

# plot the sample
#fig = plt.figure
#plt.imshow(image, cmap='Blues')
#plt.show()
plt.legend(['1','0.9','0.8','0.7','0.6','0.5','0.4','0.3','0.2','0.1'], loc='lower right')
plt.show()