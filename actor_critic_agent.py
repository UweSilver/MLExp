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
    return numpy.full(w.shape, x.transpose())

def sum_prime_b(b):
    return numpy.ones(b.shape)

def gh(x, a_max, a_min):
    return (a_max + a_min) / 2.0 + (a_max - a_min) / 2.0 * numpy.tanh(x)

def gh_prime(x, a_max, a_min):
    return (a_max - a_min) / (2.0 * (numpy.cosh(x)**2))

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
        self.actor_gamma = 0.99
        self.critic_gamma = 0.99
        self.actor_alpha = 0.003
        self.critic_alpha = 0.003
        self.sigma = 0.1
        self.actor_W_0 = numpy.random.randn(self.mid_h_dim, self.actor_h0_dim)* 0.00000001
        self.actor_W_1 = numpy.random.randn(self.actor_h2_dim, self.mid_h_dim)* 0.00000001
        self.actor_b_0 = numpy.random.randn(self.mid_h_dim, 1)* 0.00000001
        self.actor_b_1 = numpy.random.randn(self.actor_h2_dim, 1)* 0.00000001
        self.critic_W_0 = numpy.random.randn(self.mid_h_dim, self.critic_h0_dim)* 0.00000001
        self.critic_W_1 = numpy.random.randn(self.mid_h_dim, self.mid_h_dim)* 0.00000001
        self.critic_W_2 = numpy.random.randn(self.critic_h3_dim, self.mid_h_dim)* 0.00000001
        self.critic_b_0 = numpy.random.randn(self.mid_h_dim, 1)* 0.00000001
        self.critic_b_1 = numpy.random.randn(self.mid_h_dim, 1)* 0.00000001
        self.critic_b_2 = numpy.random.randn( self.critic_h3_dim, 1) * 0.00000001

        #method 0
        self.actor_W_bar_0 = self.actor_W_0
        self.actor_W_bar_1 = self.actor_W_1
        self.actor_b_bar_0 = self.actor_b_0
        self.actor_b_bar_1 = self.actor_b_1
        self.critic_W_bar_0 = self.critic_W_0
        self.critic_W_bar_1 = self.critic_W_1
        self.critic_W_bar_2 = self.critic_W_2
        self.critic_b_bar_0 = self.critic_b_0
        self.critic_b_bar_1 = self.critic_b_1
        self.critic_b_bar_2 = self.critic_b_2

        #method 3
        self.critic_W_sub_0 = numpy.random.randn(self.mid_h_dim, self.critic_h0_dim)* 0.00000001
        self.critic_W_sub_1 = numpy.random.randn(self.mid_h_dim, self.mid_h_dim)* 0.00000001
        self.critic_W_sub_2 = numpy.random.randn(self.critic_h3_dim, self.mid_h_dim)* 0.00000001
        self.critic_b_sub_0 = numpy.random.randn(self.mid_h_dim, 1)* 0.00000001
        self.critic_b_sub_1 = numpy.random.randn(self.mid_h_dim, 1)* 0.00000001
        self.critic_b_sub_2 = numpy.random.randn( self.critic_h3_dim, 1) * 0.00000001

        self.action_max = 2.0
        self.action_min = -2.0
        self.buffer = []
        self.batch_size = 256
        self.texpl = 100

        self.method0 = False
        self.tau = 0.005

        self.method1 = False
        self.sigma_target = 0.2
        self.c = 0.5

        self.method2 = False
        self.d = 2
        self.t = 0

        self.method3 = False
        return
    
    def Qohm(self, state, action):
        h0 = numpy.concatenate((numpy.array(state).reshape((3, 1)), numpy.array(action).reshape(1, 1))).reshape((self.critic_h0_dim, 1))
        input_sum_0 = self.critic_W_0 @ h0 + self.critic_b_0
        h1 = relu(input_sum_0)
        input_sum_1 = self.critic_W_1 @ h1 + self.critic_b_1
        h2 = relu(input_sum_1)
        input_sum_2 = self.critic_W_2 @ h2 + self.critic_b_2
        h3 = input_sum_2
        self.critic_result = (h0, input_sum_0, 
                              h1, input_sum_1, 
                              h2, input_sum_2, h3)
        return h3
    
    def Qohm_bar(self, state, action):
        h0 = numpy.concatenate((numpy.array(state).reshape((3, 1)), numpy.array(action).reshape(1, 1))).reshape((self.critic_h0_dim, 1))
        input_sum_0 = self.critic_W_bar_0 @ h0 + self.critic_b_bar_0
        h1 = relu(input_sum_0)
        input_sum_1 = self.critic_W_bar_1 @ h1 + self.critic_b_bar_1
        h2 = relu(input_sum_1)
        input_sum_2 = self.critic_W_bar_2 @ h2 + self.critic_b_bar_2
        h3 = input_sum_2
        self.critic_bar_result = (h0, input_sum_0, 
                              h1, input_sum_1, 
                              h2, input_sum_2, h3)
        return h3
    
    def Qohm_sub(self, state, action):
        h0 = numpy.concatenate((numpy.array(state).reshape((3, 1)), numpy.array(action).reshape(1, 1))).reshape((self.critic_h0_dim, 1))
        input_sum_0 = self.critic_W_sub_0 @ h0 + self.critic_b_sub_0
        h1 = relu(input_sum_0)
        input_sum_1 = self.critic_W_sub_1 @ h1 + self.critic_b_sub_1
        h2 = relu(input_sum_1)
        input_sum_2 = self.critic_W_sub_2 @ h2 + self.critic_b_sub_2
        h3 = input_sum_2
        self.critic_sub_result = (h0, input_sum_0, 
                              h1, input_sum_1, 
                              h2, input_sum_2, h3)
        return h3
    
    def Pithete(self, state):
        h0 = numpy.array(state).reshape((3, 1))
        input_sum_0 = self.actor_W_0 @ h0
        h1 = relu(input_sum_0)
        input_sum_1 = self.actor_W_1 @ h1
        h2 = gh(input_sum_1, self.action_max, self.action_min)
        self.actor_result = (h0, input_sum_0, h1, input_sum_1, h2)
        return h2
    
    def Pithete_bar(self, state):
        h0 = numpy.array(state).reshape((3, 1))
        input_sum_0 = self.actor_W_bar_0 @ h0
        h1 = relu(input_sum_0)
        input_sum_1 = self.actor_W_bar_1 @ h1
        h2 = gh(input_sum_1, self.action_max, self.action_min)
        self.actor_bar_result = (h0, input_sum_0, h1, input_sum_1, h2)
        return h2
    
    def select_action(self, state):
        action = self.Pithete(state)
        return action.tolist()
    
    def select_random_action(self, state):
        action = [random.random() - 0.5 * 4.0]
        h0 = numpy.array(state).reshape((3, 1))
        input_sum_0 = self.actor_W_0 @ h0
        h1 = relu(input_sum_0)
        input_sum_1 = self.actor_W_1 @ h1
        h2 = gh(input_sum_1, self.action_max, self.action_min)
        self.actor_result = (h0, input_sum_0, h1, input_sum_1, h2)
        return action

    def select_exploratory_action(self, state):
        if(len(self.buffer) < self.texpl):
            return self.select_random_action(state)
        else:
            return (numpy.clip(numpy.array(self.select_action(state)) + numpy.random.normal(0, self.sigma), self.action_min, self.action_max)).tolist()

    def q_loss (self, ohm, delta_i):
        return (delta_i - ohm)**2.0
    
    def q_loss_prime(self, ohm, delta_i):
        return 2.0 * (delta_i - ohm)
    
    def train(self, state, action, next_state, reward, done):
        h = (state, action, next_state, reward, done)
        self.buffer.append(h)

        if(len(self.buffer) < self.batch_size):
            return
        
        batch = random.sample(self.buffer, self.batch_size)
        critic_grad_2 = 0
        critic_grad_sub_2 = 0
        for (state, action, next_state, reward, done) in batch:
            critic_delta = 0
            if self.method0:
                a = self.Pithete_bar(next_state)
            else:
                a = self.Pithete(next_state)

            if self.method1:
                a = numpy.clip(a + numpy.clip(numpy.random.normal(0, self.sigma_target), -self.c, self.c), self.action_min, self.action_max)
                
            if self.method0:
                value = self.Qohm_bar(next_state, a)
                if self.method3:
                    sub = self.Qohm_sub(next_state, a)
                    if value[0] > sub[0]:
                        value = sub
                
                critic_delta = reward + (1 - done) * self.critic_gamma * value
            else:
                a = self.Pithete(next_state)
                value = self.Qohm(next_state, a)
                if self.method3:
                    sub = self.Qohm_sub(next_state, a)
                    if value[0] > sub[0]:
                        value = sub
                critic_delta = reward + (1 - done) * self.critic_gamma * value

            if self.method0:
                critic_grad_2 = critic_grad_2 + self.q_loss_prime(critic_delta, self.critic_bar_result[6]) 
                if self.method3:
                    critic_grad_sub_2 = critic_grad_sub_2 + self.q_loss_prime(critic_delta, self.critic_sub_result[6])
            else:
                critic_grad_2 = critic_grad_2 + self.q_loss_prime(critic_delta, self.critic_result[6]) 
                if self.method3:
                    critic_grad_sub_2 = critic_grad_sub_2 + self.q_loss_prime(critic_delta, self.critic_sub_result[6])

        critic_grad_2 = critic_grad_2 / self.batch_size
        if self.method3:
            critic_grad_sub_2 = critic_grad_sub_2 / self.batch_size

        if self.method0:
            (h0, input_sum_0, h1, input_sum_1, h2, input_sum_2, h3) = self.critic_bar_result
        else:
            (h0, input_sum_0, h1, input_sum_1, h2, input_sum_2, h3) = self.critic_result
        mid = (critic_grad_2)
        critic_grad_2_x = sum_prime_x(self.critic_W_2, input_sum_2).transpose() @  mid
        critic_grad_2_w = sum_prime_w(input_sum_2, self.critic_W_2) *  mid
        critic_grad_2_b = sum_prime_b(self.critic_b_2) *  mid
        
        mid =     (relu_prime(input_sum_1) * critic_grad_2_x)
        critic_grad_1_x = sum_prime_x(self.critic_W_1, h1).transpose() @ mid
        critic_grad_1_w = sum_prime_w(h1, self.critic_W_1) *  mid
        critic_grad_1_b = sum_prime_b(self.critic_b_1) * mid
        
        mid = (relu_prime(input_sum_0) * critic_grad_1_x)
        critic_grad_0_w = sum_prime_w(h0, self.critic_W_0)*  numpy.full(self.critic_W_0.shape, mid) 
        critic_grad_0_b = sum_prime_b(self.critic_b_0) *  mid


        self.critic_W_2 = self.critic_W_2 - self.critic_alpha * critic_grad_2_w
        self.critic_b_2 = self.critic_b_2 - self.critic_alpha * critic_grad_2_b
        self.critic_W_1 = self.critic_W_1 - self.critic_alpha * critic_grad_1_w
        self.critic_b_1 = self.critic_b_1 - self.critic_alpha * critic_grad_1_b
        self.critic_W_0 = self.critic_W_0 - self.critic_alpha * critic_grad_0_w
        self.critic_b_0 = self.critic_b_0 - self.critic_alpha * critic_grad_0_b

        if self.method3:
            (h0, input_sum_0, h1, input_sum_1, h2, input_sum_2, h3) = self.critic_sub_result
            mid = (critic_grad_sub_2)
            critic_grad_sub_2_x = sum_prime_x(self.critic_W_sub_2, input_sum_2).transpose() @  mid
            critic_grad_sub_2_w = sum_prime_w(input_sum_2, self.critic_W_sub_2) *  mid
            critic_grad_sub_2_b = sum_prime_b(self.critic_b_sub_2) *  mid

            mid =     (relu_prime(input_sum_1) * critic_grad_sub_2_x)
            critic_grad_sub_1_x = sum_prime_x(self.critic_W_sub_1, h1).transpose() @ mid
            critic_grad_sub_1_w = sum_prime_w(h1, self.critic_W_sub_1) *  mid
            critic_grad_sub_1_b = sum_prime_b(self.critic_b_sub_1) * mid

            mid = (relu_prime(input_sum_0) * critic_grad_sub_1_x)
            critic_grad_sub_0_w = sum_prime_w(h0, self.critic_W_sub_0)*  numpy.full(self.critic_W_sub_0.shape, mid)
            critic_grad_sub_0_b = sum_prime_b(self.critic_b_sub_0) *  mid

            self.critic_W_sub_2 = self.critic_W_sub_2 - self.critic_alpha * critic_grad_sub_2_w
            self.critic_b_sub_2 = self.critic_b_sub_2 - self.critic_alpha * critic_grad_sub_2_b
            self.critic_W_sub_1 = self.critic_W_sub_1 - self.critic_alpha * critic_grad_sub_1_w
            self.critic_b_sub_1 = self.critic_b_sub_1 - self.critic_alpha * critic_grad_sub_1_b
            self.critic_W_sub_0 = self.critic_W_sub_0 - self.critic_alpha * critic_grad_sub_0_w
            self.critic_b_sub_0 = self.critic_b_sub_0 - self.critic_alpha * critic_grad_sub_0_b
        
        if self.method2:
            self.t = self.t + 1
            if self.t % self.d == 0:
                return


        if self.method0:
            self.critic_W_bar_0 = self.critic_W_bar_0 * (1 - self.tau) + self.critic_W_0 * self.tau
            self.critic_W_bar_1 = self.critic_W_bar_1 * (1 - self.tau) + self.critic_W_1 * self.tau
            self.critic_W_bar_2 = self.critic_W_bar_2 * (1 - self.tau) + self.critic_W_2 * self.tau
            self.critic_b_bar_0 = self.critic_b_bar_0 * (1 - self.tau) + self.critic_b_0 * self.tau
            self.critic_b_bar_1 = self.critic_b_bar_1 * (1 - self.tau) + self.critic_b_1 * self.tau
            self.critic_b_bar_2 = self.critic_b_bar_2 * (1 - self.tau) + self.critic_b_2 * self.tau

        (h0, input_sum_0, h1, input_sum_1, h2) = self.actor_result
        actor_grad_1 = self.Qohm(state, h2).reshape((1, 1))
        
        mid = (gh_prime(input_sum_1, self.action_max, self.action_min) * actor_grad_1)
        actor_grad_1_x = sum_prime_x(self.actor_W_1, h2).transpose() @ mid
        actor_grad_1_w = sum_prime_w(h1, self.actor_W_1)* mid
        actor_grad_1_b = sum_prime_b(self.actor_b_1) * mid

        actor_grad_0_w = sum_prime_w(h0, self.actor_W_0) * numpy.full(self.actor_W_0.shape, (relu_prime(input_sum_0).reshape((256, 1))*actor_grad_1_x))
        actor_grad_0_b = sum_prime_b(self.actor_b_0) * ( relu_prime(input_sum_0) *actor_grad_1_x)

        self.actor_W_1 = self.actor_W_1 - self.actor_alpha * actor_grad_1_w
        self.actor_b_1 = self.actor_b_1 - self.actor_alpha * actor_grad_1_b
        self.actor_W_0 = self.actor_W_0 - self.actor_alpha * actor_grad_0_w
        self.actor_b_0 = self.actor_b_0 - self.actor_alpha * actor_grad_0_b

        if self.method0:
            self.actor_W_bar_0 = self.actor_W_bar_0 * (1 - self.tau) + self.actor_W_0 * self.tau
            self.actor_W_bar_1 = self.actor_W_bar_1 * (1 - self.tau) + self.actor_W_1 * self.tau
            self.actor_b_bar_0 = self.actor_b_bar_0 * (1 - self.tau) + self.actor_b_0 * self.tau
            self.actor_b_bar_1 = self.actor_b_bar_1 * (1 - self.tau) + self.actor_b_1 * self.tau

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

        if self.method0:
            numpy.savetxt(path + "_actor_W_bar_0", self.actor_W_bar_0)
            numpy.savetxt(path + "_actor_W_bar_1", self.actor_W_bar_1)
            numpy.savetxt(path + "_actor_b_bar_0", self.actor_b_bar_0)
            numpy.savetxt(path + "_actor_b_bar_1", self.actor_b_bar_1)
            numpy.savetxt(path + "_critic_W_bar_0", self.critic_W_bar_0)
            numpy.savetxt(path + "_critic_W_bar_1", self.critic_W_bar_1)
            numpy.savetxt(path + "_critic_W_bar_2", self.critic_W_bar_2)
            numpy.savetxt(path + "_critic_b_bar_0", self.critic_b_bar_0)
            numpy.savetxt(path + "_critic_b_bar_1", self.critic_b_bar_1)
            numpy.savetxt(path + "_critic_b_bar_2", self.critic_b_bar_2)

        if self.method3:
            numpy.savetxt(path + "_critic_W_sub_0", self.critic_W_sub_0)
            numpy.savetxt(path + "_critic_W_sub_1", self.critic_W_sub_1)
            numpy.savetxt(path + "_critic_W_sub_2", self.critic_W_sub_2)
            numpy.savetxt(path + "_critic_b_sub_0", self.critic_b_sub_0)
            numpy.savetxt(path + "_critic_b_sub_1", self.critic_b_sub_1)
            numpy.savetxt(path + "_critic_b_sub_2", self.critic_b_sub_2)

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

        if self.method0:
            self.actor_W_bar_0 = numpy.loadtxt(path + "_actor_W_bar_0")
            self.actor_W_bar_1 = numpy.loadtxt(path + "_actor_W_bar_1")
            self.actor_b_bar_0 = numpy.loadtxt(path + "_actor_b_bar_0")
            self.actor_b_bar_1 = numpy.loadtxt(path + "_actor_b_bar_1")
            self.critic_W_bar_0 = numpy.loadtxt(path + "_critic_W_bar_0")
            self.critic_W_bar_1 = numpy.loadtxt(path + "_critic_W_bar_1")
            self.critic_W_bar_2 = numpy.loadtxt(path + "_critic_W_bar_2")
            self.critic_b_bar_0 = numpy.loadtxt(path + "_critic_b_bar_0")
            self.critic_b_bar_1 = numpy.loadtxt(path + "_critic_b_bar_1")
            self.critic_b_bar_2 = numpy.loadtxt(path + "_critic_b_bar_2")

        if self.method3:
            self.critic_W_sub_0 = numpy.loadtxt(path + "_critic_W_sub_0")
            self.critic_W_sub_1 = numpy.loadtxt(path + "_critic_W_sub_1")
            self.critic_W_sub_2 = numpy.loadtxt(path + "_critic_W_sub_2")
            self.critic_b_sub_0 = numpy.loadtxt(path + "_critic_b_sub_0")
            self.critic_b_sub_1 = numpy.loadtxt(path + "_critic_b_sub_1")
            self.critic_b_sub_2 = numpy.loadtxt(path + "_critic_b_sub_2")
        pass
    
