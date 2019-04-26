from keras.models import Model
from network.custom_blocks import *
from keras.optimizers import Adam
from network.losses import *


class Gan:

    def __init__(self, input_shape, image_shape):
        self.input_shape = input_shape
        self.image_shape = image_shape
        self.lrD = 2e-4
        self.lrG = 1e-4

        # Define networks
        self.generator = self.Generator(input_shape=self.input_shape)

        # Create discriminators
        self.discriminator = self.Discriminator(image_shape=self.image_shape)

        # Define variables
        self.distorted, self.fake, self.mask, self.path, \
        self.path_mask, self.path_abgr, self.path_bgr = self.define_variables(generator=self.generator)

        self.real = Input(shape=self.input_shape)

    def Generator(self, input_shape):
        """
                Generator function creates generator model.
            :param input_shape:
            :return: model
        """
        # # #######################
        # # ## Build encoder
        # # #######################
        encoder_inputs = Input(shape=input_shape)
        x = Conv2D(64, kernel_size=5, use_bias=False, padding="same")(encoder_inputs)
        x = conv_block(128)(x)
        x = conv_block(256)(x)
        x = self_attn_block(x, 256)
        x = conv_block(512)(x)
        x = self_attn_block(x, 512)
        x = conv_block(1024)(x)

        activ_map_size = input_shape[0] // 16
        while activ_map_size > 4:
            x = conv_block(1024)(x)
            activ_map_size = activ_map_size // 2

        x = Dense(1024)(Flatten()(x))
        x = Dense(4 * 4 * 1024)(x)
        x = Reshape((4, 4, 1024))(x)
        encoder_output = upscale(512)(x)

        # # #######################
        # # ## Build decoder
        # # #######################

        x = upscale(256)(encoder_output)
        x = upscale(128)(x)
        x = self_attn_block(x, 128)
        x = upscale(64)(x)
        x = res_block(x, 64)
        x = self_attn_block(x, 64)

        outputs = []
        activ_map_size = input_shape[0] * 8
        while activ_map_size < 128:
            outputs.append(Conv2D(3, kernel_size=5, padding='same', activation="tanh")(x))
            x = upscale(64)(x)
            x = conv_block(64, strides=1)(x)
            activ_map_size *= 2

        alpha = Conv2D(1, kernel_size=5, padding='same', activation="sigmoid")(x)
        bgr = Conv2D(3, kernel_size=5, padding='same', activation="tanh")(x)
        out = concatenate([alpha, bgr])
        outputs.append(out)
        decoder_output = outputs

        # Build and compile
        generator = Model(inputs=encoder_inputs, outputs=decoder_output)

        print(generator.summary())

        return generator

    def Discriminator(self, image_shape):

        inputs = Input(shape=image_shape)

        x = dis_layer(inputs, 128)
        x = dis_layer(x, 256)
        x = dis_layer(x, 512)
        x = self_attn_block(x, 512)

        activ_map_size = image_shape[0] // 8
        while activ_map_size > 8:
            x = dis_layer(x, 256)
            x = self_attn_block(x, 256)
            activ_map_size = activ_map_size // 2

        out = Conv2D(1, kernel_size=3, padding="same")(x)

        discriminator = Model(inputs, out)
        print(discriminator.summary())

        return discriminator

    @staticmethod
    def define_variables(generator):
        distorted_input = generator.inputs[0]
        fake_output = generator.outputs[-1]
        alpha = Lambda(lambda x: x[:, :, :, :1])(fake_output)
        bgr = Lambda(lambda x: x[:, :, :, 1:])(fake_output)

        masked_fake_output = alpha * bgr + (1 - alpha) * distorted_input

        fn_generate = K.function([distorted_input], [masked_fake_output])   # Return almost like input
        fn_mask = K.function([distorted_input], [concatenate([alpha, alpha, alpha])]) # Return black image
        fn_abgr = K.function([distorted_input], [concatenate([alpha, bgr])]) # Return 4-channels maksed image
        fn_bgr = K.function([distorted_input], [bgr])   # Return input + mask

        return distorted_input, fake_output, alpha, fn_generate, fn_mask, fn_abgr, fn_bgr

    def build_train_functions(self):

        # Adversarial loss
        loss_dis, loss_adv_gen = adversarial_loss(self.discriminator, self.real, self.fake, self.distorted)
        loss_gen = loss_adv_gen

        # Alpha mask loss
        loss_gen += 1e-2 * K.mean(K.abs(self.mask))

        # Alpha mask total variation loss
        loss_gen += 0.1 * K.mean(first_order(self.mask, axis=1))
        loss_gen += 0.1 * K.mean(first_order(self.mask, axis=2))

        # L2 weight decay
        # https://github.com/keras-team/keras/issues/2662
        for loss_tensor in self.generator.losses:
            loss_gen += loss_tensor
        for loss_tensor in self.discriminator.losses:
            loss_dis += loss_tensor

        weights_dis = self.discriminator.trainable_weights
        weights_gen = self.generator.trainable_weights

        # Define training functions
        lr_factor = 1
        training_updates = Adam(lr=self.lrD * lr_factor, beta_1=0.5).get_updates(weights_dis, [], loss_dis)
        self.net_dis_train = K.function([self.distorted, self.real], [loss_dis], training_updates)
        training_updates = Adam(lr=self.lrG * lr_factor, beta_1=0.5).get_updates(weights_gen, [], loss_gen)
        self.net_gen_train = K.function([self.distorted, self.real], [loss_gen, loss_adv_gen], training_updates)

    def load_weights(self, path="data/models"):
        self.generator.load_weights("{path}/generator.h5".format(path=path))
        self.discriminator.load_weights("{path}/discriminator.h5".format(path=path))
        print("Model weights files are successfully loaded.")

    def save_weights(self, path="data/models"):
        self.generator.save_weights("{path}/generator.h5".format(path=path))
        self.discriminator.save_weights("{path}/discriminator.h5".format(path=path))
        print("Model weights files have been saved to {path}.".format(path=path))

    def train_generator(self, X, Y):

        err_gen = self.net_gen_train([X, Y])

        return err_gen

    def train_discriminator(self, X, Y):

        err_dis = self.net_dis_train([X, Y])

        return err_dis

