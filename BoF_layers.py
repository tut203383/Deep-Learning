from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
#from sklearn.cluster import KMeans
#from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
from tensorflow.keras.initializers import Constant

class BoF_Pooling(Layer):
    """
    Implements the CBoF pooling
    """

    def __init__(self, n_codewords=100, spatial_level=0,**kwargs):
        """
        Initializes a BoF Pooling layer
        :param n_codewords: the number of the codewords to be used
        :param spatial_level: 0 -> no spatial pooling, 1 -> spatial pooling at level 1 (4 regions). Note that the
         codebook is shared between the different spatial regions
        :param kwargs:
        """

        self.N_k = n_codewords
        self.spatial_level = spatial_level
        self.V, self.sigmas = None, None
        super(BoF_Pooling, self).__init__(**kwargs)

    def build(self, input_shape):
        self.V = self.add_weight(name='codebook', shape=(1, 1, input_shape[3], self.N_k), initializer='uniform',
                                 trainable=True)

        self.sigmas = self.add_weight(name='sigmas', shape=(1, 1, 1, self.N_k), initializer=Constant(0.1),
                                      trainable=True)
        super(BoF_Pooling, self).build(input_shape)

    def call(self, x):

        # Calculate the pairwise distances between the codewords and the feature vectors
        x_square = K.sum(x ** 2, axis=3, keepdims=True)
        y_square = K.sum(self.V ** 2, axis=2, keepdims=True)
        dists = x_square + y_square - 2 * K.conv2d(x, self.V, strides=(1, 1), padding='valid')
        dists = K.maximum(dists,K.epsilon())

        # Quantize the feature vectors
        quantized_features = K.softmax(- dists / (self.sigmas ** 2))

        # Compile the histogram
        if self.spatial_level == 0:
            histogram = K.mean(quantized_features, [1, 2])
        elif self.spatial_level == 1:
            shape = K.shape(quantized_features)
            mid_1 = K.cast(shape[1] / 2, 'int32')
            mid_2 = K.cast(shape[2] / 2, 'int32')
            histogram1 = K.mean(quantized_features[:, :mid_1, :mid_2, :], [1, 2])
            histogram2 = K.mean(quantized_features[:, mid_1:, :mid_2, :], [1, 2])
            histogram3 = K.mean(quantized_features[:, :mid_1, mid_2:, :], [1, 2])
            histogram4 = K.mean(quantized_features[:, mid_1:, mid_2:, :], [1, 2])
            histogram = K.stack([histogram1, histogram2, histogram3, histogram4], 1)
            histogram = K.reshape(histogram, (-1, 4 * self.N_k))
        else:
            # No other spatial level is currently supported (it is trivial to extend the code)
            assert False

        # Simple trick to avoid rescaling issues



        return histogram * self.N_k

    def compute_output_shape(self, input_shape):
        if self.spatial_level == 0:
            return (input_shape[0], self.N_k)
        elif self.spatial_level == 1:
            return (input_shape[0], 4 * self.N_k)







class BoF_Pooling_attention_hist(Layer):
    """
    Implements the CBoF pooling
    """

    def __init__(self, n_codewords=100, spatial_level=0,attention = 0, **kwargs):
        """
        Initializes a BoF Pooling layer
        :param n_codewords: the number of the codewords to be used
        :param spatial_level: 0 -> no spatial pooling, 1 -> spatial pooling at level 1 (4 regions). Note that the
         codebook is shared between the different spatial regions
        :param kwargs:
        """

        self.N_k = n_codewords
        self.spatial_level = spatial_level
        self.V, self.sigmas,self.attentionweights,self.lam = None, None,None,None
        self.attention = attention
        print(type(self) )
        super(BoF_Pooling_attention_hist, self).__init__(**kwargs)

    def build(self, input_shape):
        self.V = self.add_weight(name='codebook', shape=(1, 1, input_shape[3], self.N_k), initializer='uniform',
                                 trainable=True)
        self.attentionweights = self.add_weight(name='attentionweights', shape=(self.N_k, self.N_k), initializer='uniform',
                                                trainable=True)
        self.sigmas = self.add_weight(name='sigmas', shape=(1, 1, 1, self.N_k), initializer=Constant(0.1),
                                      trainable=True)
        self.lam = self.add_weight(name='lam', shape=(1,), initializer=Constant(0.5),
                                   trainable=True)
        super(BoF_Pooling_attention_hist, self).build(input_shape)

    def call(self, x):

        # Calculate the pairwise distances between the codewords and the feature vectors
        x_square = K.sum(x ** 2, axis=3, keepdims=True)
        y_square = K.sum(self.V ** 2, axis=2, keepdims=True)
        dists = x_square + y_square - 2 * K.conv2d(x, self.V, strides=(1, 1), padding='valid')
        dists = K.maximum(dists, 0)

        # Quantize the feature vectors
        quantized_features = K.softmax(- dists / (self.sigmas ** 2))

        # Compile the histogram
        if self.spatial_level == 0:
            histogram = K.mean(quantized_features, [1, 2])
        elif self.spatial_level == 1:
            shape = K.shape(quantized_features)
            mid_1 = K.cast(shape[1] / 2, 'int32')
            mid_2 = K.cast(shape[2] / 2, 'int32')
            histogram1 = K.mean(quantized_features[:, :mid_1, :mid_2, :], [1, 2])
            histogram2 = K.mean(quantized_features[:, mid_1:, :mid_2, :], [1, 2])
            histogram3 = K.mean(quantized_features[:, :mid_1, mid_2:, :], [1, 2])
            histogram4 = K.mean(quantized_features[:, mid_1:, mid_2:, :], [1, 2])
            histogram = K.stack([histogram1, histogram2, histogram3, histogram4], 1)
            histogram = K.reshape(histogram, (-1, 4 * self.N_k))
        else:
            # No other spatial level is currently supported (it is trivial to extend the code)
            assert False

        # Simple trick to avoid rescaling issues

        # K.dot(attentionmask,histogram)* self.N_k
        attentionmask = K.dot(histogram,self.attentionweights)
        attentionmask = K.softmax(attentionmask)
        return self.lam*(attentionmask*histogram * self.N_k) + (1 - self.lam)*histogram * self.N_k

    def compute_output_shape(self, input_shape):
        if self.spatial_level == 0:
            return (input_shape[0], self.N_k)
        elif self.spatial_level == 1:
            return (input_shape[0], 4 * self.N_k)








class BoF_Pooling_attentionbefore(Layer):
    """
    Implements the CBoF pooling
    """

    def __init__(self, n_codewords=100, spatial_level=0,attention = 0, **kwargs):
        """
        Initializes a BoF Pooling layer
        :param n_codewords: the number of the codewords to be used
        :param spatial_level: 0 -> no spatial pooling, 1 -> spatial pooling at level 1 (4 regions). Note that the
         codebook is shared between the different spatial regions
        :param kwargs:
        """

        self.N_k = n_codewords
        self.spatial_level = spatial_level
        self.V, self.sigmas,self.attentionweights,self.lam  = None, None,None,None
        self.attention = attention
        print(type(self) )
        super(BoF_Pooling_attentionbefore, self).__init__(**kwargs)

    def build(self, input_shape):
        self.V = self.add_weight(name='codebook', shape=(1, 1, input_shape[3], self.N_k), initializer='uniform',
                                 trainable=True)
        self.attentionweights = self.add_weight(name='attentionweights', shape=( input_shape[1], input_shape[2],input_shape[3],1), initializer='uniform',
                                                trainable=True)
        self.sigmas = self.add_weight(name='sigmas', shape=(1, 1, 1, self.N_k), initializer=Constant(0.1),
                                      trainable=True)
        self.lam = self.add_weight(name='lam', shape=(1,), initializer=Constant(0.5), trainable=True)
        super(BoF_Pooling_attentionbefore, self).build(input_shape)

    def call(self, x):
        attentionmask = K.conv2d(x,self.attentionweights)
        attentionmask = K.sigmoid(attentionmask)
        Xatten = self.lam *x*attentionmask  + (1-self.lam) * x


        # Calculate the pairwise distances between the codewords and the feature vectors
        x_square = K.sum(Xatten ** 2, axis=3, keepdims=True)
        y_square = K.sum(self.V ** 2, axis=2, keepdims=True)
        dists = x_square + y_square - 2 * K.conv2d(Xatten, self.V, strides=(1, 1), padding='valid')
        dists = K.maximum(dists, 0)

        # Quantize the feature vectors
        quantized_features = K.softmax(- dists / (self.sigmas ** 2))

        # Compile the histogram
        if self.spatial_level == 0:
            histogram = K.mean(quantized_features, [1, 2])
        elif self.spatial_level == 1:
            shape = K.shape(quantized_features)
            mid_1 = K.cast(shape[1] / 2, 'int32')
            mid_2 = K.cast(shape[2] / 2, 'int32')
            histogram1 = K.mean(quantized_features[:, :mid_1, :mid_2, :], [1, 2])
            histogram2 = K.mean(quantized_features[:, mid_1:, :mid_2, :], [1, 2])
            histogram3 = K.mean(quantized_features[:, :mid_1, mid_2:, :], [1, 2])
            histogram4 = K.mean(quantized_features[:, mid_1:, mid_2:, :], [1, 2])
            histogram = K.stack([histogram1, histogram2, histogram3, histogram4], 1)
            histogram = K.reshape(histogram, (-1, 4 * self.N_k))
        else:
            # No other spatial level is currently supported (it is trivial to extend the code)
            assert False

        # Simple trick to avoid rescaling issues


        return  histogram * self.N_k

    def compute_output_shape(self, input_shape):
        if self.spatial_level == 0:
            return (input_shape[0], self.N_k)
        elif self.spatial_level == 1:
            return (input_shape[0], 4 * self.N_k)

























