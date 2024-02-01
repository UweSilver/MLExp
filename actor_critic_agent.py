from agent import Agent
import numpy
import math
import random

def relu(x):
    x_copy = x.copy()
    for i in range(0, numpy.shape(x)[0]):
        x_copy[i] = max(x[i], 0)
    return x_copy

def relu_prime(x):
    x_copy = x.copy()
    for i in range(0, numpy.shape(x)[0]):
        if(x[i] > 0):
            x_copy[i] = 1
        else:
            x_copy[i] = 0
    return x_copy

def sum(x, w, b):
    return w * x + b

def sum_prime_x(w):
    return w

def sum_prime_w(x):
    return x

def sum_prime_b():
    return 1

def gh(x, a_max, a_min):
    return (a_max + a_min) / 2.0 + (a_max - a_min) / 2.0 * math.tanh(x)

def gh_prime(x, a_max, a_min):
    return (a_max - a_min) / 2.0 / (math.cosh(x)**2)

class ActorCriticAgent(Agent):
    def __init__(self) -> None:
        super().__init__()
        self.actor_h0_dim = 3
        self.actor_h2_dim = 1
        self.critic_h0_dim = self.actor_h0_dim + self.actor_h2_dim
        self.critic_h3_dim = 1
        self.mid_h_dim = 256
        self.actor_H = 2
        self.critic_H = 3
        self.actor_gamma = 0.1
        self.critic_gamma = 0.1
        self.actor_W_0 = numpy.zeros((self.actor_h0_dim, self.mid_h_dim))
        self.actor_W_1 = numpy.zeros((self.mid_h_dim, self.actor_h2_dim))
        self.actor_b_0 = numpy.zeros((self.mid_h_dim))
        self.actor_b_1 = numpy.zeros((self.actor_h2_dim))
        self.critic_W_0 = numpy.zeros((self.critic_h0_dim, self.mid_h_dim))
        self.critic_W_1 = numpy.zeros((self.mid_h_dim, self.mid_h_dim))
        self.critic_W_2 = numpy.zeros((self.mid_h_dim, self.critic_h3_dim))
        self.critic_b_0 = numpy.zeros((self.mid_h_dim))
        self.critic_b_1 = numpy.zeros((self.mid_h_dim))
        self.critic_b_2 = numpy.zeros((self.critic_h3_dim))
        self.action_max = 2.0
        self.action_min = -2.0
        print("W: " + str(self.W))
        return
    
    def Qohm(self, state, action):
        h0 = numpy.concatenate((state, action))
        input_sum_0 = self.critic_W_0 * h0 + self.critic_b_0
        h1 = relu(input_sum_0)
        input_sum_1 = self.critic_W_1 * h1 + self.critic_b_1
        h2 = relu(input_sum_1)
        input_sum_2 = self.critic_W_2 * h2 + self.critic_b_2
        h3 = relu(input_sum_2, self.action_max, self.action_min)
        self.qritic_result = (h0, input_sum_0, h1, input_sum_1, h2, input_sum_2, h3)
        return h3
    
    def Pithete(self, state):
        h0 = state
        input_sum_0 = self.actor_W_0 * h0
        h1 = relu(input_sum_0)
        input_sum_1 = self.actor_W_1 * h1
        h2 = input_sum_1
        self.actor_result = (h0, input_sum_0, h1, input_sum_1, h2)
        return h2
    
    def select_action(self, state):
        action = self.Pithete(state)
        return action

    def select_exploratory_action(self, state):
        dice = random.random()
        if dice > self.epsilon:
            return self.select_action(state)
        else:
            return (random.random() * (self.action_max - self.action_min) + self.action_min)

    def q_loss (self, ohm, delta_i):
        return (delta_i - ohm)**2
    
    def q_loss_prime(self, ohm, delta_i):
        return 2 * (delta_i - ohm)
    
    def train(self, state, action, next_state, reward, done):
        h = (state, action, next_state, reward, done)
        critic_delta = reward + (1 - done) * self.critic_gamma * self.Qohm(next_state, self.select_action(next_state))

        (h0, input_sum_0, h1, input_sum_1, h2, input_sum_2, h3) = self.critic_result
        critic_grad_2 = self.q_loss_prime(critic_delta, h3)
        critic_grad_2_x = critic_grad_2 * relu_prime(input_sum_2) * sum_prime_x(self.critic_W_2)
        critic_grad_2_w = critic_grad_2 * relu_prime(input_sum_2) * sum_prime_w(h2)
        critic_grad_2_b = critic_grad_2 * relu_prime(input_sum_2) * sum_prime_b()

        critic_grad_1_x = critic_grad_2_x * relu_prime(input_sum_1) * sum_prime_x(self.critic_W_1)
        critic_grad_1_w = critic_grad_2_x * relu_prime(input_sum_1) * sum_prime_w(h1)
        critic_grad_1_b = critic_grad_2_x * relu_prime(input_sum_1) * sum_prime_b()

        critic_grad_0_x = critic_grad_1_x * relu_prime(input_sum_0) * sum_prime_x(self.critic_W_0)
        critic_grad_0_w = critic_grad_1_x * relu_prime(input_sum_0) * sum_prime_w(h0)
        critic_grad_0_b = critic_grad_1_x * relu_prime(input_sum_0) * sum_prime_b()

        self.critic_W_2 = self.critic_W_2 - self.critic_alpha * critic_grad_2_w
        self.critic_b_2 = self.critic_b_2 - self.critic_alpha * critic_grad_2_b
        self.critic_W_1 = self.critic_W_1 - self.critic_alpha * critic_grad_1_w
        self.critic_b_1 = self.critic_b_1 - self.critic_alpha * critic_grad_1_b
        self.critic_W_0 = self.critic_W_0 - self.critic_alpha * critic_grad_0_w
        self.critic_b_0 = self.critic_b_0 - self.critic_alpha * critic_grad_0_b

        (h0, input_sum_0, h1, input_sum_1, h2) = self.actor_result
        actor_grad_1 = self.Qohm(state, h2)
        actor_grad_1_x = actor_grad_1  * sum_prime_x(self.actor_W_1)
        actor_grad_1_w = actor_grad_1  * sum_prime_w(h1)
        actor_grad_1_b = actor_grad_1  * sum_prime_b()

        actor_grad_0_x = actor_grad_1_x * relu_prime(input_sum_0) * sum_prime_x(self.actor_W_0)
        actor_grad_0_w = actor_grad_1_x * relu_prime(input_sum_0) * sum_prime_w(h0)
        actor_grad_0_b = actor_grad_1_x * relu_prime(input_sum_0) * sum_prime_b()

        self.actor_W_1 = self.actor_W_1 - self.actor_alpha * actor_grad_1_w
        self.actor_b_1 = self.actor_b_1 - self.actor_alpha * actor_grad_1_b
        self.actor_W_0 = self.actor_W_0 - self.actor_alpha * actor_grad_0_w
        self.actor_b_0 = self.actor_b_0 - self.actor_alpha * actor_grad_0_b

        pass

    def save_models(self, path):
        pass

    def load_models(self, path):
        pass
    
