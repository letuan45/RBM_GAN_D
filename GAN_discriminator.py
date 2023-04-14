from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Input
from keras.layers import LeakyReLU
from tensorflow import keras

# define the standalone supervised and unsupervised discriminator models
def define_discriminator(in_shape):
	# Khai báo các layers
	inpt = Input(shape=in_shape)
	conv1 = Conv2D(filters=32, kernel_size=(3,3), strides=(2,2), padding='same')(inpt)
	act_leak1 = LeakyReLU(alpha=0.3)(conv1)
	dropout = Dropout(0.4)(act_leak1)
	conv2 = Conv2D(filters=64, kernel_size=(3,3), strides=(2,2), padding='same')(dropout)
	act_leak2 = LeakyReLU(alpha=0.3)(conv2)
	flat = Flatten()(act_leak2)
	den = Dense(1, activation='sigmoid')(flat)
  	# Khởi tạo model
	model = keras.models.Model(inputs = [inpt], outputs = [den])

	# Compile với optimizer
	opt = keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

	return model