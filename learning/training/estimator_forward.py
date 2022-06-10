import os
import tfquaternion as tfq
import tensorflow as tf
import numpy as np

class DeepTracker(tf.estimator.Estimator):
    def __init__(self, feature_columns, initial_learning_rate=0.001, hidden_units=(20,20), model_dir='.', norm_data=None):
        model_params = {"initial_learning_rate": initial_learning_rate, "feature_columns": feature_columns, 'hidden_units': hidden_units, 'norm_data': norm_data}
        super().__init__(
            model_fn=self.tracker_model_fn,
            params=model_params,
            model_dir=model_dir)

    @staticmethod
    def tracker_model_fn(features, labels, mode, params):
        """Model function for Estimator."""
        input_layer = tf.feature_column.input_layer(
            features=features, feature_columns=params['feature_columns'])

        # Connect the first hidden layer to input layer
        # (features["x"]) with relu activation
        prev_layer = input_layer
        # prev_layer = tf.layers.dropout(inputs=prev_layer, rate=0.15, training=mode == tf.estimator.ModeKeys.TRAIN)
        for (layer,num_units) in enumerate(params['hidden_units']):
            # prev_layer = tf.layers.batch_normalization(prev_layer)
            tf.summary.histogram("layer_%d" % layer, prev_layer)
            prev_layer = tf.layers.dense(prev_layer, num_units, activation=tf.nn.leaky_relu,
                                         )#kernel_regularizer=tf.contrib.layers.l2_regularizer(.000005),
                                         #bias_regularizer=tf.contrib.layers.l2_regularizer(.000005))

            # prev_layer += input_layer
            # if layer == 1:
            #     prev_layer = tf.layers.dropout(inputs=prev_layer, rate=0.2, training=mode == tf.estimator.ModeKeys.TRAIN)

        # prev_layer = tf.layers.batch_normalization(prev_layer)
        tf.summary.histogram("final_layer", prev_layer)
        residual_hat = tf.layers.dense(prev_layer, 9)


        # Provide an estimator spec for `ModeKeys.PREDICT`.
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions={'m': residual_hat})

        m_actual = labels['m']
        m_dipole = labels['md']
        residual = m_dipole - m_actual

        tf.summary.histogram("m_hat", residual_hat)
        tf.summary.histogram("m", residual)

        eval_metric_ops = {}

        m_err = residual - residual_hat
        m_err_sq = tf.reduce_sum(m_err ** 2, axis=1)
        m_err_mse = tf.reduce_mean(m_err_sq)
        tf.summary.histogram("m_err", tf.sqrt(m_err_sq))
        tf.summary.scalar('m_err_rmse', tf.sqrt(m_err_mse))
        tf.summary.scalar('m_err_percent_rms', tf.sqrt(tf.reduce_mean((m_err / labels['m'])**2)))
        eval_metric_ops["m_err_rmse"] = tf.metrics.mean(tf.sqrt(m_err_mse))
        eval_metric_ops["m_err_percent_rms"] = tf.metrics.mean(tf.sqrt(tf.reduce_mean((m_err / labels['m'])**2)))


        loss_inverse = m_err_mse

        loss = loss_inverse

        learning_rate = tf.train.exponential_decay(params["initial_learning_rate"], tf.train.get_global_step(),
                                                   5000, 0.9, staircase=True)

        tf.summary.scalar("learning_rate", learning_rate)
        # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=.9)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

        # Provide an estimator spec for `ModeKeys.EVAL` and `ModeKeys.TRAIN` modes.
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op,
            predictions={'m': residual}, eval_metric_ops=eval_metric_ops)
