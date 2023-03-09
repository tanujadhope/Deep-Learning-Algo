# Deep-Learning-Algo


Introduction:
AlexNet was designed by Hinton, winner of the 2012 ImageNet competition, and his student Alex Krizhevsky. It was also after that year that more and deeper neural networks were proposed, such as the excellent vgg, GoogleLeNet. Its official data model has an accuracy rate of 57.1% and top 1-5 reaches 80.2%. This is already quite outstanding for traditional machine learning classification algorithms.
Size / Operation	Filter	  Depth	  Stride	Padding	    Number of Parameters	             Forward Computation
3* 227 * 227						
Conv1 + Relu	    11 * 11	  96	     4		              (11*11*3 + 1) * 96=34944	          (11*11*3 + 1) * 96 * 55 * 55=105705600
96 * 55 * 55						
Max Pooling	      3 * 3		            2			
96 * 27 * 27						
Norm						
Conv2 + Relu	    5 * 5	    256     	1     	2	        (5 * 5 * 96 + 1) * 256=614656   	  (5 * 5 * 96 + 1) * 256 * 27 * 27=448084224
256 * 27 * 27						
Max Pooling	      3 * 3		            2			
256 * 13 * 13						
Norm						
Conv3 + Relu    	3 * 3	    384	      1	      1	          (3 * 3 * 256 + 1) * 384=885120	  (3 * 3 * 256 + 1) * 384 * 13 * 13=149585280
384 * 13 * 13						
Conv4 + Relu	    3 * 3	    384     	1	      1	          (3 * 3 * 384 + 1) * 384=1327488	  (3 * 3 * 384 + 1) * 384 * 13 * 13=224345472
384 * 13 * 13						
Conv5 + Relu	    3 * 3	    256     	1	      1	            (3 * 3 * 384 + 1) * 256=884992	(3 * 3 * 384 + 1) * 256 * 13 * 13=149563648
256 * 13 * 13						
Max Pooling     	3 * 3		            2			
256 * 6 * 6						
Dropout (rate 0.5)						
FC6 + Relu				                                          	256 * 6 * 6 * 4096=37748736	      256 * 6 * 6 * 4096=37748736
4096						
Dropout (rate 0.5)						
FC7 + Relu				                                            4096 * 4096=16777216	              4096 * 4096=16777216
4096						
FC8 + Relu					                                         4096 * 1000=4096000	                  4096 * 1000=4096000
1000 classes						
Overall					            62369152=62.3 million	1135906176=1.1 billion
Conv VS FC				      	Conv:3.7million (6%) , FC: 58.6 million (94% )	Conv: 1.08 billion (95%) , FC: 58.6 million (5%)



Why does AlexNet achieve better results?
Relu activation function is used.
Relu function: f (x) = max (0, x)

ReLU-based deep convolutional networks are trained several times faster than tanh and sigmoid- based networks. The following figure shows the number of iterations for a four-layer convolutional network based on CIFAR-10 that reached 25% training error in tanh and ReLU:

Standardization ( Local Response Normalization )
After using ReLU f (x) = max (0, x), you will find that the value after the activation function has no range like the tanh and sigmoid functions, so a normalization will usually be done after ReLU, and the LRU is a steady proposal (Not sure here, it should be proposed?) One method in neuroscience is called "Lateral inhibition", which talks about the effect of active neurons on its surrounding neurons.

Dropout
Dropout is also a concept often said, which can effectively prevent overfitting of neural networks. Compared to the general linear model, a regular method is used to prevent the model from overfitting. In the neural network, Dropout is implemented by modifying the structure of the neural network itself. For a certain layer of neurons, randomly delete some neurons with a defined probability, while keeping the individuals of the input layer and output layer neurons unchanged, and then update the parameters according to the learning method of the neural network. In the next iteration, rerandom Remove some neurons until the end of training.

Enhanced Data ( Data Augmentation )
In deep learning, when the amount of data is not large enough, there are generally 4 solutions:

Data augmentation- artificially increase the size of the training set-create a batch of "new" data from existing data by means of translation, flipping, noise

Regularization——The relatively small amount of data will cause the model to overfit, making the training error small and the test error particularly large. By adding a regular term after the Loss Function , the overfitting can be suppressed. The disadvantage is that a need is introduced Manually adjusted hyper-parameter.

Dropout- also a regularization method. But different from the above, it is achieved by randomly setting the output of some neurons to zero

Unsupervised Pre-training- use Auto-Encoder or RBM's convolution form to do unsupervised pre-training layer by layer, and finally add a classification layer to do supervised Fine-Tuning
