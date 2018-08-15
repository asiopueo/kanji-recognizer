



def classifier(image):
	import brainstorm as bs
	import numpy as np
	from PIL import Image, ImageOps

	network = bs.Network.from_hdf5('C1_best_network.hdf5')

	image.thumbnail((64,64), Image.ANTIALIAS)
	image = ImageOps.fit(image, (63,64))
	#image = ImageOps.invert(image)
	#image.show()
	#data = np.array(image).reshape(image.size[0], image.size[1], 3).dot([0.2, 0.7, 0.1]).reshape(image.size[0], image.size[1], 1) / 255
	data = np.array(image).reshape(image.size[0], image.size[1], 1) * 200

	network.provide_external_data({'features': np.array([[data]])}, all_inputs=False)
	network.forward_pass(training_pass=False)

	classification = network.get('Output.outputs.predictions')[0][0]
	print np.argmax(classification)
	print classification
	print "\n Sum = ", sum(classification)


if __name__ == '__main__':
	import sys
	from PIL import Image
	
	fileName = sys.argv[1]
	image = Image.open(fileName)
	
	classifier(image)











