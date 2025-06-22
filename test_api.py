from pettingzoo.test import parallel_api_test
from custom_environment import CustomEnvironment
env = CustomEnvironment()

observations, _ = env.reset()


parallel_api_test(env, num_cycles=1000)
