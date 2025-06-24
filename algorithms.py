import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os
from datetime import datetime
import jsonlines
import os
from datetime import datetime
import jsonlines
from gym import spaces
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
RAY_COUNT = 15
ALL_TYPE_COUNT = 6  # none, out_of_bounds, predator, River, Rock, Field
RAY_FEATURE_DIM = 4 + ALL_TYPE_COUNT

MOVE_BIN_VALUES = [-2.0, -0.5, 0.0, 0.5, 2.0]
ATK_BIN_VALUES = [-1.0, -0.25, 0.0, 0.25, 1.0]
SPIN_BIN_VALUES = [-1.0,  0.0, 1.0]

observation_space = spaces.Dict({
    "rays": spaces.Box(
        low=0, high=1.0,
        shape=(RAY_COUNT, RAY_FEATURE_DIM),
        dtype=np.float32
    ),
    "facing": spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
    "velocity": spaces.Box(low=-5.0, high=5.0, shape=(2,), dtype=np.float32),
    "good_vector": spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
    "good_distance": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
    "good_info": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
    "bad_vector": spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
    "bad_distance": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
})

action_space = spaces.Dict({
    "x": spaces.Discrete(len(MOVE_BIN_VALUES)),
    "y": spaces.Discrete(len(MOVE_BIN_VALUES)),
    "spin": spaces.Discrete(len(SPIN_BIN_VALUES))
})


class AgentNet(models.Model):
    def __init__(self, total_steps=5_000_000, log_dir=None):
        super(AgentNet, self).__init__()

        # flat feature encoder
        self.flat_input = layers.Dense(32, activation='relu')
        self.ray_input_proj = layers.Dense(
            128, activation='relu', name="ray_input_proj")
        self.ray2 = layers.Dense(
            96, activation='relu', name="ray2_proj")  # for ray features
        # Normalization layers for input encodings
        self.flat_norm = layers.LayerNormalization(name="flat_norm")
        self.ray_norm = layers.LayerNormalization(name="ray_norm")

        self.lstm = layers.LSTM(256, return_sequences=True, return_state=True,
                                dropout=0.0, recurrent_dropout=0.1)
        self.ln = layers.LayerNormalization()

        # policy & value heads
        self.policy_dense = layers.Dense(64, activation='relu')

        self.x_dir = layers.Dense(
            len(MOVE_BIN_VALUES))
        self.y_dir = layers.Dense(
            len(MOVE_BIN_VALUES))
        self.spin = layers.Dense(
            len(SPIN_BIN_VALUES))

        self.value_dense = layers.Dense(32, activation='relu')
        self.value_out = layers.Dense(1, activation='linear')
        # Layer normalization for policy and value feature heads
        self.policy_norm = layers.LayerNormalization(name="policy_norm")
        self.value_norm = layers.LayerNormalization(name="value_norm")

        self.lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=1e-5,
            decay_steps=total_steps,
            end_learning_rate=1e-6,
            power=1.0)

        self.entropy_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=0.01,
            decay_steps=total_steps,
            end_learning_rate=0.001,
            power=1.0)

        self.global_step = tf.Variable(
            0, dtype=tf.int64, trainable=False, name="global_step")

        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.lr_schedule)

        self.metrics_path = os.path.join(log_dir, "train_metrics.jsonl")

    @tf.function
    def call(self, flat_seq, rays_seq, h1, c1, training=False):
        B = tf.shape(flat_seq)[0]
        T = tf.shape(flat_seq)[1]

        # encode flat inputs
        flat_enc = self.flat_input(flat_seq)
        flat_enc = self.flat_norm(flat_enc)
        # build a boolean mask that is True for real timesteps (non‑padding)
        mask = tf.math.not_equal(tf.reduce_sum(tf.abs(flat_seq), axis=-1), 0)

        # encode rays inputs simply
        rays_flat = tf.reshape(rays_seq, [B, T, RAY_COUNT * RAY_FEATURE_DIM])
        rays_enc = self.ray_input_proj(rays_flat)
        rays_enc = self.ray2(rays_enc)
        rays_enc = self.ray_norm(rays_enc)
        combined = tf.concat([flat_enc, rays_enc], axis=-1)

        o1, nh1, nc1 = self.lstm(
            combined, mask=mask, initial_state=[h1, c1], training=training)

        o1 = self.ln(o1)

        # policy & value heads
        pf = self.policy_dense(o1)
        pf = self.policy_norm(pf)

        dist_x = tfd.Categorical(logits=self.x_dir(pf))
        dist_y = tfd.Categorical(logits=self.y_dir(pf))
        dist_spin = tfd.Categorical(logits=self.spin(pf))

        vf = self.value_dense(o1)
        vf = self.value_norm(vf)
        out = {
            "value":         self.value_out(vf),
            "lstm_state":    [nh1, nc1],
            "dist_x":       dist_x,
            "dist_y":       dist_y,
            "dist_spin":    dist_spin,
        }
        return out

    def select_action(self, obs, hidden_state):
        flat, rays = zip(*obs)
        flat = tf.expand_dims(flat, 0)
        rays = tf.expand_dims(rays, 0)

        if hidden_state is None:
            h1, c1 = self.get_initial_state(batch_size=1)
        else:
            h1, c1 = hidden_state

        outputs = self.call(flat, rays, h1, c1, training=False)

        dist_x = outputs["dist_x"]
        dist_y = outputs["dist_y"]
        dist_spin = outputs["dist_spin"]

        a_x = dist_x.sample()
        a_y = dist_y.sample()
        a_spin = dist_spin.sample()

        logp_tot = (dist_x.log_prob(a_x) +
                    dist_y.log_prob(a_y) +
                    dist_spin.log_prob(a_spin))
        return {
            "value": float(outputs["value"][0, 0, 0].numpy()),
            "x_dir": a_x.numpy()[0][0],
            "y_dir": a_y.numpy()[0][0],
            "spin": a_spin.numpy()[0][0],
            "lstm_state": outputs["lstm_state"],
            "logp_total": logp_tot.numpy()[0],
        }

    def train_one_epoch(self,
                        flat_b, rays_b,
                        x_b, y_b, spin_b,
                        old_logp_b, old_val_b, ret_b, adv_b,
                        s_h1, s_c1,
                        clip_eps):
        # mask out padded timesteps
        mask = tf.cast(tf.math.not_equal(tf.reduce_sum(
            tf.abs(flat_b), axis=-1), 0), tf.float32)  # shape [B, T]
        with tf.GradientTape() as tape:
            outputs = self.call(flat_b, rays_b, s_h1, s_c1, training=True)
            # raw and old value predictions for clipping
            raw_values = tf.squeeze(outputs["value"], axis=-1)  # shape [B, T]
            old_values = tf.stop_gradient(raw_values)
            values = raw_values

            dist_x = outputs["dist_x"]
            dist_y = outputs["dist_y"]
            dist_spin = outputs["dist_spin"]

            lp_x = dist_x.log_prob(x_b)
            lp_y = dist_y.log_prob(y_b)
            lp_spin = dist_spin.log_prob(spin_b)
            lp_tot = lp_x + lp_y + lp_spin

            log_ratio = lp_tot - old_logp_b
            ratio = tf.exp(log_ratio)
            clipped_ratio = tf.clip_by_value(
                ratio, 1.0 - clip_eps, 1.0 + clip_eps)

            surrogate = tf.where(
                adv_b > 0,
                tf.minimum(ratio * adv_b, clipped_ratio * adv_b),
                tf.maximum(ratio * adv_b, clipped_ratio * adv_b)
            )
            policy_loss = - \
                tf.reduce_sum(mask * surrogate) / tf.reduce_sum(mask)

            # normalize returns
            ret_mean = tf.reduce_mean(ret_b)
            ret_std = tf.math.reduce_std(ret_b) + 1e-8
            ret_b_norm = (ret_b - ret_mean) / ret_std

            # clipped value update
            clipped_values = old_values + \
                tf.clip_by_value(values - old_values, -clip_eps, clip_eps)
            clipped_values_norm = (clipped_values - ret_mean) / ret_std

            value_loss = tf.reduce_sum(
                mask * tf.square(ret_b_norm - clipped_values_norm)) / tf.reduce_sum(mask)
            entropy = tf.reduce_mean(dist_x.entropy()
                                     + dist_y.entropy()
                                     + dist_spin.entropy())
            tf.print("POLICY step", self.optimizer.iterations,
                     "mean|adv|=", tf.reduce_mean(tf.abs(adv_b)),
                     "policy_loss=", policy_loss,
                     "value_loss=", value_loss,
                     "entropy=", entropy)
            # use optimizer.iterations to sync schedules
            entropy_coef = self.entropy_schedule(self.optimizer.iterations)
            loss = policy_loss + 0.5 * value_loss - entropy_coef * entropy
            grads = tape.gradient(loss, self.trainable_variables)

            clipped_grads, _ = tf.clip_by_global_norm(grads, 3)
            grad_norm = tf.linalg.global_norm(grads)
            self.optimizer.apply_gradients(
                zip(clipped_grads, self.trainable_variables))
            # ---- JSONL metric logging ----
            grad_norm = tf.linalg.global_norm(clipped_grads)
            metrics = {
                "step": int(self.global_step.numpy()),
                "loss_total": float(loss.numpy()),
                "loss_policy": float(policy_loss.numpy()),
                "loss_value": float(value_loss.numpy()),
                "entropy_mean": float(entropy.numpy()),
                "entropy_coef": float(entropy_coef.numpy()),
                "grad_norm": float(grad_norm.numpy())
            }
            with jsonlines.open(self.metrics_path, "a") as jf:
                jf.write(metrics)
            self.global_step.assign_add(1)

    def optimize_model(self, buffer, hidden_states, gamma=0.99, lam=0.95, clip_eps=0.1, epochs=3, batch_size=128):

        buffer_size = len(buffer)
        flats = [[step[0] for step in seq] for seq in buffer]
        rays = [[step[1] for step in seq] for seq in buffer]
        actions = [[step[2] for step in seq] for seq in buffer]
        log_probs = [[step[3] for step in seq] for seq in buffer]
        rewards = [[step[4] for step in seq] for seq in buffer]
        dones = [[step[5] for step in seq] for seq in buffer]
        values = [[step[6] for step in seq] for seq in buffer]
        initial_lstm_states = [seq for seq in hidden_states]
        flat_list = tf.ragged.constant(
            flats, dtype=tf.float32, ragged_rank=1).to_tensor()
        rays_list = tf.ragged.constant(
            rays, dtype=tf.float32, ragged_rank=1).to_tensor()
        x_bin = tf.ragged.constant(
            [[a["x_dir"] for a in seq] for seq in actions],
            dtype=tf.int32, ragged_rank=1).to_tensor()
        y_bin = tf.ragged.constant(
            [[a["y_dir"] for a in seq] for seq in actions],
            dtype=tf.int32, ragged_rank=1).to_tensor()
        spin_bin = tf.ragged.constant(
            [[a["spin"] for a in seq] for seq in actions],
            dtype=tf.int32, ragged_rank=1).to_tensor()
        old_log_probs = tf.ragged.constant(
            [[float(tf.squeeze(lp).numpy()) for lp in seq]
             for seq in log_probs],
            dtype=tf.float32, ragged_rank=1
        ).to_tensor()

        rewards = tf.ragged.constant(
            rewards, dtype=tf.float32, ragged_rank=1).to_tensor()
        reward_mean = tf.reduce_mean(rewards)
        reward_std = tf.math.reduce_std(rewards) + 1e-8

        # scale rewards
        rewards = (rewards - reward_mean) / reward_std

        dones = tf.ragged.constant(
            [[float(d) for d in seq] for seq in dones],
            dtype=tf.float32, ragged_rank=1).to_tensor()
        values = tf.ragged.constant(
            [[float(tf.squeeze(v).numpy()) for v in seq] for seq in values],
            dtype=tf.float32, ragged_rank=1
        ).to_tensor()
        values = (values - reward_mean) / reward_std

        B = tf.shape(flat_list)[0]
        T = tf.shape(flat_list)[1]

        advantages = tf.TensorArray(tf.float32, size=T)
        returns = tf.TensorArray(tf.float32, size=T)
        gae = tf.zeros([B], dtype=tf.float32)
        next_value = tf.zeros([B], dtype=tf.float32)

        for t in range(T - 1, -1, -1):
            delta = rewards[:, t] + gamma * next_value * \
                (1 - dones[:, t]) - values[:, t]
            gae = delta + gamma * lam * (1 - dones[:, t]) * gae
            advantages = advantages.write(t, tf.reshape(gae, [B]))
            returns = returns.write(t, tf.reshape(gae + values[:, t], [B]))
            next_value = values[:, t]

        adv_tensor = tf.transpose(advantages.stack(), [1, 0])
        returns_tensor = tf.transpose(returns.stack(), [1, 0])

        lstm_h = tf.stack([tf.squeeze(s[0], axis=0)
                           for s in initial_lstm_states], axis=0)
        lstm_c = tf.stack([tf.squeeze(s[1], axis=0)
                           for s in initial_lstm_states], axis=0)
        ds = tf.data.Dataset.from_tensor_slices({
            "flat":       flat_list,
            "rays":       rays_list,
            "x":          x_bin,
            "y":          y_bin,
            "spin":       spin_bin,
            "old_logp":   old_log_probs,
            "old_val":    values,
            "returns":    returns_tensor,
            "advantages": adv_tensor,
            "h1":         lstm_h,
            "c1":         lstm_c,
        }).shuffle(buffer_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)

        clip_eps_t = tf.convert_to_tensor(clip_eps, dtype=tf.float32)
        for _ in range(epochs):
            for batch in ds:
                self.train_one_epoch(
                    batch["flat"], batch["rays"],
                    batch["x"], batch["y"], batch["spin"],
                    batch["old_logp"], batch["old_val"], batch["returns"], batch["advantages"],
                    batch["h1"], batch["c1"],
                    clip_eps_t
                )

    def get_initial_state(self, batch_size=1):
        return [
            tf.zeros((batch_size, 256), dtype=tf.float32),
            tf.zeros((batch_size, 256), dtype=tf.float32),
        ]


def get_model(log_dir=None):
    model = AgentNet(log_dir=log_dir)
    dummy_flat = tf.zeros((1, 64, 12), dtype=tf.float32)
    dummy_rays = tf.zeros(
        (1, 64, RAY_COUNT, RAY_FEATURE_DIM), dtype=tf.float32)
    h1, c1 = model.get_initial_state(batch_size=1)
    model(dummy_flat, dummy_rays, h1, c1, training=False)
    return model


def load_model(filepath=None, model=None, log_dir=None):
    new_model = AgentNet(log_dir=log_dir)
    dummy_flat = tf.zeros((1, 64, 12), dtype=tf.float32)
    dummy_rays = tf.zeros(
        (1, 64, RAY_COUNT, RAY_FEATURE_DIM), dtype=tf.float32)
    h1, c1 = new_model.get_initial_state(batch_size=1)
    new_model(dummy_flat, dummy_rays, h1, c1, training=False)
    print("Loading model weights from:",
          filepath if filepath else "provided model")
    if filepath is not None:
        new_model.load_weights(filepath)
    elif model is not None:
        new_model.set_weights(model.get_weights())
    return new_model
