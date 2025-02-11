from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet_v2 import FullyConnectedNetwork
from ray.rllib.utils import try_import_tf
tf = try_import_tf()


class PpoModel(TFModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        super().__init__(obs_space, action_space, num_outputs, model_config,
                         name)
        self.inputs = tf.keras.layers.Input(shape=obs_space.shape,
                                            name="vector_observation")
        x = tf.keras.layers.Dense(
            128,
            activation=tf.keras.activation.swish,
            kernel_initializer=tf.keras.initializers.VarianceScaling(1.0),
            bias_initializer=tf.keras.initializers.Zeros())(self.inputs)
        x = tf.keras.layers.Dense(
            128,
            activation=tf.keras.activation.swish,
            kernel_initializer=tf.keras.initializers.VarianceScaling(1.0),
            bias_initializer=tf.keras.initializers.Zeros())(x)
        value_out = tf.keras.layers.Dense(1)(x)

        act_space = list(action_space.nvec)
        branches_logit = [
            tf.keras.layers.Dense(
                branch,
                use_bias=False,
                kernel_initializer=tf.keras.initializers.VarianceScaling(0.01),
                bias_initializer=tf.keras.initializers.Zeros())(x)
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