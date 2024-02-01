from agent import Agent
import numpy
import math
import random

def relu(x):
    x_copy = x.copy()
    for i in range(0, numpy.shape(x)[0]):
        x_copy[i] = numpy.max(x[i], 0)
    return x_copy

def relu_prime(x):
    x_copy = x.copy()
    for i in range(0, numpy.shape(x)[0]):
        if(x[i] > 0):
            x_copy[i] = 1.0
        else:
            x_copy[i] = 0.0
    return x_copy

def sum(x, w, b):
    return w * x + b

def sum_prime_x(w, x):
    return w

def sum_prime_w(x, w):
    # print("sum_prime_w")
    # print(x.shape)
    # print(w.shape)
    # print(numpy.full(w.shape, x.transpose()).shape)
    return numpy.full(w.shape, x.transpose())

def sum_prime_b(b):
    return numpy.ones(b.shape)

def gh(x, a_max, a_min):
    return (a_max + a_min) / 2.0 + (a_max - a_min) / 2.0 * numpy.tanh(x)

def gh_prime(x, a_max, a_min):
    return (a_max - a_min) / 2.0 / (numpy.cosh(x)**2)

class ActorCriticAgent(Agent):
    def __init__(self, seed) -> None:
        random.seed(seed)
        numpy.random.seed(seed=seed)
        self.actor_h0_dim = 3
        self.actor_h2_dim = 1
        self.critic_h0_dim = self.actor_h0_dim + self.actor_h2_dim
        self.critic_h3_dim = 1
        self.mid_h_dim = 256
        self.actor_H = 2
        self.critic_H = 3
        self.actor_gamma = 0.1
        self.critic_gamma = 0.1
        self.actor_alpha = 0.003
        self.critic_alpha = 0.003
        self.sigma = 0.1
        self.actor_W_0 = numpy.zeros((self.mid_h_dim, self.actor_h0_dim))
        self.actor_W_1 = numpy.zeros((self.actor_h2_dim, self.mid_h_dim))
        self.actor_b_0 = numpy.zeros((self.mid_h_dim, 1))
        self.actor_b_1 = numpy.zeros((self.actor_h2_dim, 1))
        self.critic_W_0 = numpy.zeros((self.mid_h_dim, self.critic_h0_dim))
        self.critic_W_1 = numpy.zeros((self.mid_h_dim, self.mid_h_dim))
        self.critic_W_2 = numpy.zeros((self.critic_h3_dim, self.mid_h_dim))
        self.critic_b_0 = numpy.zeros(( self.mid_h_dim, 1))
        self.critic_b_1 = numpy.zeros((self.mid_h_dim, 1))
        self.critic_b_2 = numpy.zeros(( self.critic_h3_dim, 1))
        self.action_max = 2.0
        self.action_min = -2.0
        self.epsilon = 0.05
        self.buffer = []
        self.batch_size = 256
        self.texpl = 10000
        return
    
    def Qohm(self, state, action):
        h0 = numpy.concatenate((numpy.array(state).reshape((3, 1)), numpy.array(action).reshape(1, 1))).reshape((self.critic_h0_dim, 1))
        input_sum_0 = self.critic_W_0 @ h0 + self.critic_b_0
        h1 = relu(input_sum_0)
        input_sum_1 = self.critic_W_1 @ h1 + self.critic_b_1
        h2 = relu(input_sum_1)
        input_sum_2 = self.critic_W_2 @ h2 + self.critic_b_2
        h3 = gh(input_sum_2, self.action_max, self.action_min)
        self.critic_result = (
                                self.critic_W_0, 
                              self.critic_b_0, 
                              self.critic_W_1,
                              self.critic_b_1,
                              self.critic_W_2, 
                              self.critic_b_2, 
                              h0, input_sum_0, 
                              h1, input_sum_1, 
                              h2, input_sum_2, h3)
        return h3
    
    def Pithete(self, state):
        h0 = numpy.array(state).reshape((3, 1))
        input_sum_0 = self.actor_W_0 @ h0
        h1 = relu(input_sum_0)
        input_sum_1 = self.actor_W_1 @ h1
        h2 = input_sum_1
        self.actor_result = (self.actor_W_0, self.actor_b_0, self.actor_W_1, self.actor_b_1, h0, input_sum_0, h1, input_sum_1, h2)
        return h2
    
    def select_action(self, state):
        action = self.Pithete(state)
        return action.tolist()
    
    def select_random_action(self, state):
        action = [random.random() - 0.5 * 4.0]
        self.actor_result = (self.actor_W_0, self.actor_b_0, self.actor_W_1, self.actor_b_1, numpy.array(state).reshape((3, 1)), numpy.zeros((self.mid_h_dim, 1)), numpy.zeros((self.mid_h_dim, 1)), numpy.zeros((self.actor_h2_dim, 1)), numpy.zeros((self.actor_h2_dim, 1)))
        return action

    def select_exploratory_action(self, state):
        if(len(self.buffer) < self.texpl):
            return self.select_random_action(state)
        return (numpy.clip(numpy.array(self.select_action(state)) + numpy.random.normal(0, self.sigma), self.action_min, self.action_max)).tolist()

    def q_loss (self, ohm, delta_i):
        return (delta_i - ohm)**2.0
    
    def q_loss_prime(self, ohm, delta_i):
        return 2.0 * (delta_i - ohm)
    
    def train(self, state, action, next_state, reward, done):
        h = (state, action, next_state, reward, done, self.actor_result)
        self.buffer.append(h)

        if(len(self.buffer) < self.batch_size):
            return
        
        batch = random.sample(self.buffer, self.batch_size)
        for (state, action, next_state, reward, done, actor_result) in batch:
            critic_delta = reward + (1 - done) * self.critic_gamma * self.Qohm(next_state, action)

            (critic_W_0, critic_b_0, critic_W_1, critic_b_1, critic_W_2, critic_b_2, h0, input_sum_0, h1, input_sum_1, h2, input_sum_2, h3) = self.critic_result
            critic_grad_2 = self.q_loss_prime(critic_delta, h3)

            mid = (gh_prime(input_sum_2, self.action_max, self.action_min)* critic_grad_2)
            critic_grad_2_x = sum_prime_x(critic_W_2, h2).transpose() @  mid
            critic_grad_2_w = sum_prime_w(h2, critic_W_2) *  mid
            critic_grad_2_b = sum_prime_b(critic_b_2) *  mid
            
            mid =     (relu_prime(input_sum_1) * critic_grad_2_x)
            critic_grad_1_x = sum_prime_x(critic_W_1, h1).transpose() @ mid
            critic_grad_1_w = sum_prime_w(h1, critic_W_1) *  mid
            critic_grad_1_b = sum_prime_b(critic_b_1) * mid
            
            mid = (relu_prime(input_sum_0) * critic_grad_1_x)
            critic_grad_0_w = sum_prime_w(h0, critic_W_0)*  numpy.full(critic_W_0.shape, mid) 
            critic_grad_0_b = sum_prime_b(critic_b_0) *  mid

            self.critic_W_2 = self.critic_W_2 - self.critic_alpha * critic_grad_2_w
            self.critic_b_2 = self.critic_b_2 - self.critic_alpha * critic_grad_2_b
            self.critic_W_1 = self.critic_W_1 - self.critic_alpha * critic_grad_1_w
            self.critic_b_1 = self.critic_b_1 - self.critic_alpha * critic_grad_1_b
            self.critic_W_0 = self.critic_W_0 - self.critic_alpha * critic_grad_0_w
            self.critic_b_0 = self.critic_b_0 - self.critic_alpha * critic_grad_0_b

            (actor_W_0, actor_b_0, actor_W_1, actor_b_1, h0, input_sum_0, h1, input_sum_1, h2) = actor_result
            actor_grad_1 = self.Qohm(state, h2).reshape((1, 1))
            
            actor_grad_1_x = sum_prime_x(actor_W_1, h2).transpose() @ actor_grad_1
            actor_grad_1_w = sum_prime_w(h1, actor_W_1)* actor_grad_1 
            actor_grad_1_b = sum_prime_b(actor_b_1) * actor_grad_1

            actor_grad_0_w = sum_prime_w(h0, actor_W_0) * numpy.full(actor_W_0.shape, (relu_prime(input_sum_0).reshape((256, 1))*actor_grad_1_x))
            actor_grad_0_b = sum_prime_b(actor_b_0) * ( relu_prime(input_sum_0) *actor_grad_1_x)

            self.actor_W_1 = self.actor_W_1 - self.actor_alpha * actor_grad_1_w
            self.actor_b_1 = self.actor_b_1 - self.actor_alpha * actor_grad_1_b
            self.actor_W_0 = self.actor_W_0 - self.actor_alpha * actor_grad_0_w
            self.actor_b_0 = self.actor_b_0 - self.actor_alpha * actor_grad_0_b

        pass

    def save_models(self, path):
        numpy.savetxt(path + "_actor_W_0", self.actor_W_0)
        numpy.savetxt(path + "_actor_W_1", self.actor_W_1)
        numpy.savetxt(path + "_actor_b_0", self.actor_b_0)
        numpy.savetxt(path + "_actor_b_1", self.actor_b_1)
        numpy.savetxt(path + "_critic_W_0", self.critic_W_0)
        numpy.savetxt(path + "_critic_W_1", self.critic_W_1)
        numpy.savetxt(path + "_critic_W_2", self.critic_W_2)
        numpy.savetxt(path + "_critic_b_0", self.critic_b_0)
        numpy.savetxt(path + "_critic_b_1", self.critic_b_1)
        numpy.savetxt(path + "_critic_b_2", self.critic_b_2)

        pass

    def load_models(self, path):
        self.actor_W_0 = numpy.loadtxt(path + "_actor_W_0")
        self.actor_W_1 = numpy.loadtxt(path + "_actor_W_1")
        self.actor_b_0 = numpy.loadtxt(path + "_actor_b_0")
        self.actor_b_1 = numpy.loadtxt(path + "_actor_b_1")
        self.critic_W_0 = numpy.loadtxt(path + "_critic_W_0")
        self.critic_W_1 = numpy.loadtxt(path + "_critic_W_1")
        self.critic_W_2 = numpy.loadtxt(path + "_critic_W_2")
        self.critic_b_0 = numpy.loadtxt(path + "_critic_b_0")
        self.critic_b_1 = numpy.loadtxt(path + "_critic_b_1")
        self.critic_b_2 = numpy.loadtxt(path + "_critic_b_2")
        pass
    
