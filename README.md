# theano_artistic_cnn

Implementation of generating artistic pictures[1] in Theano[2]. We used AlexNet here, even though original work used VGG.

This code is light enough to be run on the CPU only. To enhance convergence speed, RMSprop is employed for learning rate schedules.

Trained weights are taken from [4]. 

# How to run

Prerequisite : Theano[2] and Pylearn2[3]

1) Run preprocess_img.py :
python preprocess_img.py

2) Run artistic_alexnet_inference.py :
THEANO_FLAGS='floatX=float32,device=cpu,nvcc.fastmath=True' python artistic_alexnet_inference.py

3) Run artistic_alexnet_train.py :
THEANO_FLAGS='floatX=float32,device=cpu,nvcc.fastmath=True' python artistic_alexnet_train_rmsprop.py

# References

[1] Gatys, Leon A., Alexander S. Ecker, and Matthias Bethge. "A neural algorithm of artistic style." arXiv preprint arXiv:1508.06576 (2015).

[2] http://deeplearning.net/software/theano/index.html

[3] http://deeplearning.net/software/pylearn2/

[4] https://github.com/uoguelph-mlrg/theano_alexnet

# Contact

sihyun0826@kaist.ac.kr
