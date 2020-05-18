from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet_v2 import FullyConnectedNetwork
from ray.rllib.utils import try_import_tf
tf = try_import_tf()


class CircConv(tf.keras.Model):
    def __init__(self, filters, kernel_size, activation='relu'):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv1d = tf.keras.layers.Conv1D(filters,
                                             kernel_size,
                                             padding='valid',
                                             activation=activation)

    def call(self, x, training=False):
        num_pad = self.kernel_size // 2
        inp_rs = tf.expand_dims(x, axis=-1)
        inp_padded = tf.concat(
            [inp_rs[..., -num_pad:, :], inp_rs, inp_rs[..., :num_pad, :]], -2)
        out = self.conv1d(inp_padded)
        out = tf.reshape(out, shape=[tf.shape(out)[0], 32 * 4])
        return out


class PpoModel(TFModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        super().__init__(obs_space, action_space, num_outputs, model_config,
                         name)
        self.inputs = tf.keras.layers.Input(shape=obs_space.shape,
                                            name="vector_observation")
        z = CircConv(kernel_size=3, filters=4)(self.inputs[:, 11:43])
        z0 = tf.keras.layers.Dense(128)(tf.concat([z, self.inputs[:, 0:7]],
                                                  axis=-1))
        z1 = tf.keras.layers.Dense(128)(tf.concat([z, self.inputs[:, 7:11]],
                                                  axis=-1))
        y = tf.concat([
            tf.concat([z, self.inputs[:, i:(i + 8)]], axis=-1)
            for i in range(43, 99, 8)
        ],
                      axis=0)
        z2_ = tf.keras.layers.Dense(128)(y)
        z2 = tf.concat(tf.split(z2_, 7, axis=0), axis=-1)
        x = tf.concat([z, z0, z1, z2], axis=-1)
        x = tf.keras.layers.Dense(512)(x)
        value_out = tf.keras.layers.Dense(1)(x)

        x = tf.keras.layers.Dense(256)(x)
        act_space = list(action_space.nvec)
        branches_logit = [
            tf.keras.layers.Dense(branch, activation=None, use_bias=False)(x)
            for branch in act_space
        ]
        models = [
            tf.distributions.Categorical(logits=branches_logit[i])
            for i in range(len(act_space))
        ]
        sample_op = tf.squeeze(tf.stack([model.sample(1) for model in models],
                                        axis=-1),
                               axis=0)
        pi_act = sample_op
        layer_out = tf.concat([
            tf.one_hot(tf.cast(pi_act[:, i], tf.int32), branch)
            for i, branch in enumerate(act_space)
        ],
                              axis=1)
        tf.identity(layer_out, name="one_hot_action")

        self.base_model = tf.keras.Model(self.inputs, [layer_out, value_out])
        self.register_variables(self.base_model.variables)

    def forward(self, input_dict, state, seq_lens):
        model_out, self._value_out = self.base_model(input_dict["obs"])
        return model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])