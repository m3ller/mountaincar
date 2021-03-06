import gym
import numpy as np
import tensorflow as tf

class ReinforcementLearner():
    # Environment doc: https://github.com/openai/gym/wiki/MountainCar-v0
    # Actions: [0]left push, [1]no push, [2]right push
    # Observations: [0]position, [1]velocity
    def __init__(self, environment):
        self.env = environment
        self.n_act = environment.action_space.n
        self.n_obs = environment.observation_space.shape[0]
  
    def policy_grad(self):
        # Build network
        global_step = tf.Variable(0, trainable=False)
        observation = tf.placeholder(tf.float32, [None, self.n_obs], "pg_obs")

        n_hidden = 6
        w1 = tf.get_variable("pg_w1", [self.n_obs, n_hidden])
        b1 = tf.get_variable("pg_b1", [1, n_hidden])
        w2 = tf.get_variable("pg_w2", [n_hidden, n_hidden])
        b2 = tf.get_variable("pg_b2", [1, n_hidden])
        w3 = tf.get_variable("pg_w3", [n_hidden, self.n_act])
        b3 = tf.get_variable("pg_b3", [1, self.n_act])

        # Calculate probability
        temp_logp = tf.matmul(observation, w1) + b1
        temp_logp = tf.nn.tanh(temp_logp)
        temp_logp = tf.matmul(temp_logp, w2) + b2
        temp_logp = tf.nn.relu(temp_logp)
        temp_logp = tf.matmul(temp_logp, w3) + b3
        logp = tf.nn.tanh(temp_logp)

        # Epsilon Greedy
        threshold = tf.train.exponential_decay(0.5, global_step, 100, 0.8, staircase=True)
        prob = tf.cond(tf.less(tf.random_uniform([1])[0], threshold), lambda: tf.ones(tf.shape(logp), tf.float32), lambda: logp)
        prob = tf.nn.softmax(prob)

        # Update network parameters with advantage and optimizer
        action = tf.placeholder(tf.float32, [None, self.n_act], "pg_act")
        advantage = tf.placeholder(tf.float32, [None], "pg_advantage")

        adjustment = tf.log(tf.reduce_sum(tf.multiply(tf.reshape(prob,[-1]), tf.reshape(action,[-1]))))
        adjustment = tf.multiply(tf.reshape(adjustment,[-1]), tf.reshape(advantage, [-1]))
        loss = -tf.reduce_sum(adjustment, axis=0)

        # optimizer
        optimizer = tf.train.AdamOptimizer().minimize(loss, global_step=global_step)

        # Store on TensorBoard
        smy_pg_loss = tf.summary.scalar("pg_loss", loss)
        smy_pg_prob = tf.summary.histogram("pg_prob", prob)
        smy_pg_thres = tf.summary.scalar("pg_theshold", threshold)
        smy_op = tf.summary.merge([smy_pg_loss, smy_pg_prob, smy_pg_thres])

        return observation, prob, action, advantage, optimizer, smy_op
  
    def value_grad(self):
        # Build network
        observation = tf.placeholder(tf.float32, [None, self.n_obs], "vg_obs")
        observed_value = tf.placeholder(tf.float32, [None, 1], "vg_value")

        n_hidden = 16
        w1 = tf.get_variable("vg_w1", [self.n_obs, n_hidden])
        b1 = tf.get_variable("vg_b1", [n_hidden])
        #w2 = tf.get_variable("vg_w2", [n_hidden, n_hidden])
        #b2 = tf.get_variable("vg_b2", [n_hidden])
        w3 = tf.get_variable("vg_w3", [n_hidden, 1])
        b3 = tf.get_variable("vg_b3", [1])

        temp_value = tf.matmul(observation, w1) + b1
        temp_value = tf.nn.tanh(temp_value)
        #temp_value = tf.matmul(temp_value, w2) + b2
        #temp_value = tf.nn.relu(temp_value)
        expected_value = tf.matmul(temp_value, w3) + b3

        # Calculate and improve V(s) approximation
        diffs = tf.reshape(observed_value,[-1]) - tf.reshape(expected_value, [-1])
        loss = tf.nn.l2_loss(diffs)

        # optimizer
        optimizer = tf.train.AdamOptimizer().minimize(loss)

        # Calculate the advantage
        advantages = diffs
        #advantages -= tf.reduce_mean(advantages)

        # Store on TensorBoard
        smy_vg_loss = tf.summary.scalar("vg_loss", loss)
        smy_vg_advantage = tf.summary.histogram("vg_advantage", advantages)
        summary_op = tf.summary.merge([smy_vg_loss, smy_vg_advantage])
        return observation, observed_value, optimizer, expected_value, summary_op
  
    def run_episode(self, sess, pg_obs, pg_prob, render_flag=False):
        observations = []
        actions = []
        rewards = []

        # Running policy_grad(observation) to produce actions in the game.
        observation = self.env.reset()   # initial observation
        for _ in xrange(200):
            if render_flag:
                self.env.render()

            action_prob = sess.run(pg_prob, feed_dict={pg_obs: np.expand_dims(observation, 0)})
            action = np.random.choice(range(self.n_act), 1, p=action_prob[0])[0]
            new_observation, reward, done, info = self.env.step(action)

            # Store transistions and update
            observations.append(observation)
            actions.append(action)
            add_reward = (observation[0])**2 * 10
            #print add_reward
            #add_reward = observation[0]*10
            rewards.append(reward + add_reward)
            observation = new_observation[:]

            if done:
                break

        # Convert transition lists to arrays
        observations = np.array(observations)
        rewards = np.expand_dims(np.array(rewards), 1)

        # Convert action choices to one-hots
        len_act = len(actions)
        action_onehot = np.zeros((len_act, self.n_act))
        action_onehot[range(len_act), actions] = 1

        return observations, action_onehot, rewards

    """ Update the parameters in the policy and value networks.
    Update is dependent on the observed rewards from the previous game.
    """
    def update_param(self, sess, transition_tuple, pg_obs, pg_prob, pg_action, pg_advantage, pg_optimizer, pg_summary_loss, vg_obs, vg_val, vg_optimizer, vg_advantage, vg_summary_loss):
        observations, actions, rewards = transition_tuple

        # Calculate observed reward
        gamma = 0.5
        for i in xrange(2, len(rewards)+1):
            rewards[-i] += gamma * rewards[-i+1]    # Account for future reward

        # Calculate advantage
        expected_value = sess.run(vg_advantage, feed_dict={vg_obs: observations})
        advantages = rewards - expected_value
        val_mean = np.mean(advantages)
        #advantages -= val_mean

        # Update value_grad
        _, vg_loss = sess.run([vg_optimizer, vg_summary_loss],\
                      feed_dict={vg_obs: observations, vg_val: rewards})

        # Update policy_grad
        _, pg_loss = sess.run([pg_optimizer, pg_summary_loss],\
                      feed_dict={pg_obs: observations,\
                                 pg_action:actions,\
                                 pg_advantage: advantages[0]})
        
        return vg_loss, pg_loss, sum(rewards)
  
def main():
    env = gym.make("MountainCar-v0")

    # Build network
    learner = ReinforcementLearner(env)
    pg_obs, pg_prob, pg_act, pg_adv, pg_opt, pg_sumop = learner.policy_grad()
    vg_obs, vg_val, vg_opt, vg_adv, vg_sumop = learner.value_grad()
   
    # Run episode
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_writer = tf.summary.FileWriter("./tf_logs/", sess.graph)
         
        for i in xrange(2000):
            if i% 100 == 0:
                transition_tuple = learner.run_episode(sess, pg_obs, pg_prob, True)
            else:
                transition_tuple = learner.run_episode(sess, pg_obs, pg_prob)
            vg_summary, pg_summary, total_reward = learner.update_param(sess, transition_tuple, pg_obs, pg_prob, pg_act, pg_adv, pg_opt, pg_sumop, vg_obs, vg_val, vg_opt, vg_adv, vg_sumop)

            if i % 50 == 0:
                train_writer.add_summary(vg_summary, i)
                train_writer.add_summary(pg_summary, i)
                print total_reward
         
        # Testing
        learner.run_episode(sess, pg_obs, pg_prob, True)

if __name__ == "__main__":
    main()
