from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.models import Model

def conv_block(input, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x

def encoder_block(input, num_filters):
    x = conv_block(input, num_filters)
    p = MaxPool2D((2, 2))(x)
    return x, p

def decoder_block(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x

def build_model(input_shape):
    inputs = Input(input_shape)

    s1, p1 = encoder_block(inputs, input_shape[0]/8)
    s2, p2 = encoder_block(p1, input_shape[0]/4)
    s3, p3 = encoder_block(p2, input_shape[0]/2)
    s4, p4 = encoder_block(p3, input_shape[0])

    b1 = conv_block(p4, input_shape[0]*2)

    d1 = decoder_block(b1, s4, input_shape[0])
    d2 = decoder_block(d1, s3, input_shape[0]/2)
    d3 = decoder_block(d2, s2, input_shape[0]/4)
    d4 = decoder_block(d3, s1, input_shape[0]/8)

    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)

    model = Model(inputs, outputs, name="U-Net")
    return model

if __name__ == "__main__":
    input_shape = (512, 512, 3)
    model = build_model(input_shape)
    model.summary()
