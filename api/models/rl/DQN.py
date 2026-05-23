import numpy as np
import math
import random
import traceback

class CartPoleEnv:
    """A pure-numpy physics simulator for the CartPole environment"""
    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4
        self.state = None

    def reset(self):
        self.state = np.random.uniform(low=-0.05, high=0.05, size=(4,))
        return self.state

    def step(self, action):
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        temp = (force + self.polemass_length * theta_dot**2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc

        self.state = np.array([x, x_dot, theta, theta_dot])

        done = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )
        reward = 1.0 if not done else 0.0
        return self.state, reward, done

class SimpleDQN:
    """A lightning-fast Pure Numpy Neural Network for Q-Value Approximation"""
    def __init__(self, state_dim=4, action_dim=2, lr=0.01):
        self.W1 = np.random.randn(state_dim, 24) * 0.1
        self.b1 = np.zeros(24)
        self.W2 = np.random.randn(24, action_dim) * 0.1
        self.b2 = np.zeros(action_dim)
        self.lr = lr

    def predict(self, state):
        z1 = np.dot(state, self.W1) + self.b1
        a1 = np.maximum(0, z1)  # ReLU
        q_values = np.dot(a1, self.W2) + self.b2
        return q_values, a1

    def update(self, state, target_q):
        q_values, a1 = self.predict(state)

        # Calculate loss and gradients
        dz2 = q_values - target_q
        loss = np.mean(dz2**2)

        dW2 = np.outer(a1, dz2)
        db2 = dz2
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * (a1 > 0)
        dW1 = np.outer(state, dz1)
        db1 = dz1

        # Apply gradients
        self.W1 -= self.lr * dW1
        self.W2 -= self.lr * dW2
        self.b1 -= self.lr * db1
        self.b2 -= self.lr * db2
        return loss

def run_dqn(data):
    try:
        params = data.get('parameters', {})
        episodes = int(params.get('episodes', 150))
        learning_rate = float(params.get('learningRate', 0.01))
        gamma = float(params.get('gamma', 0.95))
        epsilon_decay = float(params.get('epsilonDecay', 0.99))

        env = CartPoleEnv()
        agent = SimpleDQN(state_dim=4, action_dim=2, lr=learning_rate)

        epsilon = 1.0
        epsilon_min = 0.01

        reward_loss_history =[]
        animation_episodes =[]

        max_score = 0

        # Capture 5 evenly spaced episodes to show the gradual learning progression
        render_episodes = np.unique(np.linspace(1, episodes, 5, dtype=int)).tolist()

        for ep in range(1, episodes + 1):
            state = env.reset()
            total_reward = 0
            total_loss = 0
            steps = 0

            ep_states =[]

            # Allow the pole to balance for up to 250 steps (a few seconds of 60fps video)
            while steps < 250:
                # Save the raw physics coordinates for the React Canvas
                ep_states.append([float(v) for v in state])

                # Epsilon-Greedy Action Selection
                if np.random.rand() <= epsilon:
                    action = random.randrange(2)
                else:
                    q_values, _ = agent.predict(state)
                    action = np.argmax(q_values)

                next_state, reward, done = env.step(action)
                total_reward += reward
                steps += 1

                # Calculate Target Q
                q_values, _ = agent.predict(state)
                target_q = q_values.copy()
                if done:
                    target_q[action] = reward
                else:
                    next_q, _ = agent.predict(next_state)
                    target_q[action] = reward + gamma * np.amax(next_q)

                # Train Neural Network
                loss = agent.update(state, target_q)
                total_loss += loss

                state = next_state
                if done:
                    # Append the final failure frame so we actually see it fall
                    ep_states.append([float(v) for v in state])
                    break

            # Decay Epsilon
            if epsilon > epsilon_min:
                epsilon *= epsilon_decay

            max_score = max(max_score, total_reward)

            # Log metrics for Recharts
            avg_loss = total_loss / max(1, steps)
            reward_loss_history.append({
                "episode": ep,
                "reward": float(total_reward),
                "loss": float(avg_loss),
                "epsilon": float(epsilon)
            })

            # If this is a targeted episode, save its frame history
            if ep in render_episodes:
                # If it balanced for more than 150 steps, we consider it a success
                is_success = bool(steps >= 150)
                animation_episodes.append({
                    "episode": ep,
                    "success": is_success,
                    "score": steps,
                    "states": ep_states
                })

        return {
            'episodes': episodes,
            'max_score': float(max_score),
            'final_epsilon': float(epsilon),
            'animation_episodes': animation_episodes,
            'reward_loss_history': reward_loss_history
        }

    except Exception as e:
        return {'error': str(e), 'traceback': traceback.format_exc()}
