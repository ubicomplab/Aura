import os
import tfquaternion as tfq
import tensorflow as tf
import numpy as np

USE_QUAT = False
USE_POS = True
USE_POLAR = False


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

        input_layers = [tf.feature_column.input_layer(
            features=features, feature_columns=[tf.feature_column.numeric_column(key=('m%d' % (i+coil*3))) for i in range(3)]) for coil in range(3)]

        input_sq = [tf.pow(tf.layers.dense(rx, 3, activation=None), 2) for rx in input_layers]
        input_calib = [tf.sqrt(tf.reduce_sum(rx_sq, axis=1)) for rx_sq in input_sq]
        input_layer = (tf.stack(input_calib, axis=-1) / 20000) - .5

        # Connect the first hidden layer to input layer
        # (features["x"]) with relu activation
        prev_layer = input_layer
        # prev_layer = tf.layers.dropout(inputs=prev_layer, rate=0.15, training=mode == tf.estimator.ModeKeys.TRAIN)
        for num_units in params['hidden_units']:
            # prev_layer = tf.layers.batch_normalization(prev_layer)
            prev_layer = tf.layers.dense(prev_layer, num_units, activation=tf.nn.leaky_relu,
                                         )#kernel_regularizer=tf.contrib.layers.l2_regularizer(.000005),
                                         #bias_regularizer=tf.contrib.layers.l2_regularizer(.000005))
            # prev_layer += input_layer
            # prev_layer = tf.layers.dropout(inputs=prev_layer, rate=0.1, training=mode == tf.estimator.ModeKeys.TRAIN)

        # Connect the output layer to second hidden layer (no activation fn)
        output_pos = tf.layers.dense(prev_layer, 3)
        output_rot = tf.layers.dense(prev_layer, 4)

        if USE_POLAR:
            theta = output_pos[:,0]
            phi = output_pos[:,1]
            r = output_pos[:,2]
            tf.summary.histogram("r", r)
            tf.summary.histogram("theta", theta)
            tf.summary.histogram("phi", phi)
            output_pos = tf.transpose(tf.stack((r * tf.sin(theta) * tf.cos(phi),
                                   r * tf.sin(phi),
                                   r * tf.cos(theta) * tf.cos(phi))))

        # Provide an estimator spec for `ModeKeys.PREDICT`.
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions={'pos': output_pos, 'rot': output_rot})

        tf.summary.histogram("x", labels['pos'][:,0])
        tf.summary.histogram("y", labels['pos'][:,1])
        tf.summary.histogram("z", labels['pos'][:,2])

        try:
            scale = params['norm_data']['pos_scale'].values
        except:
            scale = params['norm_data']['pos_scale']

        tf_scale = tf.convert_to_tensor(scale, dtype=tf.float32)
        err = tf.multiply((labels['pos'] - output_pos), tf_scale)
        err_sq_m = tf.reduce_sum(err**2, axis=1)
        tf.summary.histogram("pos_error", tf.sqrt(err_sq_m)*1000)

        mse_m = tf.reduce_mean(err_sq_m)
        mse_mm = mse_m * (1000**2)
        loss_pos = mse_mm

        if USE_QUAT:
            q1 = tfq.Quaternion(output_rot)
            q1_norm =tfq.Quaternion(tf.divide(q1, tf.clip_by_value(q1.abs(), 0.0001, 1000000)))
            q2 = tfq.Quaternion(labels['rot'])
            quat_between = q1_norm * q2.conjugate()

            loss_norm = tf.reduce_mean((1-q1.norm()) ** 2)

            tf.summary.histogram("q1_norm", q1.norm())
            tf.summary.histogram("q2_norm", q2.norm())

            w = tf.squeeze(tf.slice(quat_between._q, [0, 0], [-1, 1]))

            tf.summary.histogram("w", w)

            angle_between_rad = 2*tf.acos(tf.clip_by_value(w, -1, 1))
            # angle_between_rad = 2*tf.atan2(tf.sqrt(1 - tf.clip_by_value(w**2, 0,1)), tf.clip_by_value(w, .1, 1))

            angle_between_deg = angle_between_rad * 180 / 3.14159
            tf.summary.histogram("angle_between_deg", angle_between_deg)
            mse_deg = tf.reduce_mean(angle_between_deg**2)
            rmse_deg = tf.sqrt(mse_deg)
            tf.summary.scalar("angular_error_deg", rmse_deg)
            # loss_rot = tf.reduce_mean(tf.pow(1-w, 2))
            loss_rot = mse_deg/(2**2)
            tf.summary.scalar("loss_rot", loss_rot)
            tf.summary.scalar("loss_norm", loss_norm)

        for (i, dim) in enumerate("xyz"):
            tf.summary.histogram("pos_error_mm_%s" % dim, 1000*tf.abs(labels['pos'][:,i] - output_pos[:,i]) * scale[i])
            tf.summary.scalar("pos_error_rmse_mm_%s" % dim, 1000*tf.sqrt(tf.reduce_mean((labels['pos'][:,i] - output_pos[:,i])**2)) * scale[i])


        loss = 0# loss_pos + loss_rot + loss_norm
        if USE_POS:
            loss += loss_pos
        if USE_QUAT:
            loss += loss_rot + loss_norm
        tf.summary.scalar("Position_RMSE_mm", tf.sqrt(mse_mm))
        tf.summary.scalar("loss_pos", loss_pos)

        learning_rate = tf.train.exponential_decay(params["initial_learning_rate"], tf.train.get_global_step(),
                                                   3000, 0.85, staircase=True)

        tf.summary.scalar("learning_rate", learning_rate)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

        # Provide an estimator spec for `ModeKeys.EVAL` and `ModeKeys.TRAIN` modes.
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op,
            predictions={'pos': output_pos, 'rot': output_rot}, eval_metric_ops={"Position_RMSE_mm": tf.metrics.mean(tf.sqrt(mse_mm))})
