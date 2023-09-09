from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
env = gym_super_mario_bros.make('SuperMarioBros-v0')


observation = env.reset()
done = True


if __name__  =='__main__':
    for step in range(5000000):
        if done:
            state = env.reset()
        state, reward, done, info = env.step(env.action_space.sample())
        print(env.action_space.sample())
        env.render()

    env.close()
