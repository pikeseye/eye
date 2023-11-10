import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model

def conv_block(x, num_filters):
    x = layers.Conv2D(num_filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(num_filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    return x

def encoder_block(x, num_filters):
    x = conv_block(x, num_filters)
    p = layers.MaxPool2D((2, 2))(x)
    return x, p

def attention_gate(g, s, num_filters):
    Wg = layers.Conv2D(num_filters, 1, padding="same")(g)
    Wg = layers.BatchNormalization()(Wg)

    Ws = layers.Conv2D(num_filters, 1, padding="same")(s)
    Ws = layers.BatchNormalization()(Ws)

    out = layers.Activation("relu")(Wg + Ws)
    out = layers.Conv2D(num_filters, 1, padding="same")(out)
    out = layers.Activation("sigmoid")(out)

    return out * s

def decoder_block(x, s, num_filters):
    x = layers.UpSampling2D(interpolation="bilinear")(x)
    s = attention_gate(x, s, num_filters)
    x = layers.Concatenate()([x, s])
    x = conv_block(x, num_filters)
    return x

def build_model(input_shape):
    """ Inputs """
    inputs = layers.Input(input_shape)

    """ Encoder """
    s1, p1 = encoder_block(inputs, input_shape[0]/4)
    s2, p2 = encoder_block(p1, input_shape[0]/2)
    s3, p3 = encoder_block(p2, input_shape[0])

    b1 = conv_block(p3, 512)

    """ Decoder """
    d1 = decoder_block(b1, s3, input_shape[0])
    d2 = decoder_block(d1, s2, input_shape[0]/2)
    d3 = decoder_block(d2, s1, input_shape[0]/4)

    """ Outputs """
    outputs = layers.Conv2D(1, 1, padding="same", activation="sigmoid")(d3)

    """ Model """
    model = Model(inputs, outputs, name="attention_unet")
    return model

if __name__ == "__main__":
    input_shape = (256, 256, 3)
    model = build_model(input_shape)
    model.summary()
