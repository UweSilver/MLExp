from actor_critic_agent import *
import numpy

print("hello")

# relu
vec = numpy.random.randn(10)
print(str(vec))
print(str(relu(vec)))
print(str(relu_prime(vec)))
print(str(vec))

# sum
h2 = numpy.random.randn(10)
print(str(h2))
