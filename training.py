
import h5py

import brainstorm as bs
from brainstorm.data_iterators import Minibatches
from brainstorm.handlers import PyCudaHandler

bs.global_rnd.seed(42)






DATA_FILE = "testC1_normalized.hdf5"

dataset = h5py.File(DATA_FILE, 'r')

x_tr, y_tr = dataset['training set']['features'][:], dataset['training set']['labels'][:]
x_va, y_va = dataset['validation set']['features'][:], dataset['validation set']['labels'][:]

getter_tr = Minibatches(100, default=x_tr, targets=y_tr)
getter_va = Minibatches(100, default=x_va, targets=y_va)




########### Network ##############
inp, jis = bs.tools.get_in_out_layers('classification', (63,64,1), 320, projection_name='JIS')
network = bs.Network.from_layer(
	inp >> 
	bs.layers.Dropout(drop_prob=0.2) >>
	bs.layers.Convolution2D(32, kernel_size=(3, 3), padding=2, name='ConvolutionLayer1') >>
	bs.layers.Pooling2D(type="max", kernel_size=(2, 2), stride=(2, 2)) >>
	bs.layers.Convolution2D(32, kernel_size=(3, 3), padding=2, name='ConvolutionLayer2') >>
	bs.layers.Pooling2D(type="max", kernel_size=(2, 2), stride=(2, 2)) >>
	bs.layers.Convolution2D(64, kernel_size=(3, 3), padding=2, name='ConvolutionLayer1') >>
	bs.layers.Pooling2D(type="max", kernel_size=(3, 3), stride=(2, 2)) >>
	bs.layers.FullyConnected(1200, name='HiddenLayer', activation='rel') >>
	bs.layers.Dropout(drop_prob=0.5) >>
	jis
)

network.set_handler(PyCudaHandler())
network.initialize(bs.initializers.Gaussian(0.01))
network.set_weight_modifiers({"JIS": bs.value_modifiers.ConstrainL2Norm(1)})



##################################
trainer = bs.Trainer(bs.training.MomentumStepper(learning_rate=0.1, momentum=0.9))
trainer.add_hook(bs.hooks.ProgressBar())
scorers = [bs.scorers.Accuracy(out_name='Output.outputs.predictions')]
trainer.add_hook(bs.hooks.MonitorScores('valid_getter', scorers, name='validation'))
trainer.add_hook(bs.hooks.SaveBestNetwork('validation.Accuracy', filename='C1_best_network.hdf5', name='best weights', criterion='max'))
trainer.add_hook(bs.hooks.StopAfterEpoch(30))



############ Train ###############
trainer.train(network, getter_tr, valid_getter=getter_va)
print "Best validation accuracy:", max(trainer.logs["validation"]["Accuracy"])




