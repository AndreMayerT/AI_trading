import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv
from stable_baselines3 import A2C
from envs.trading_env import TradingEnv
from sklearn.model_selection import train_test_split
import keyboard

df = pd.read_csv("15min_timeseries.csv")
df["Data"] = pd.to_datetime(df["Data"], format="%d/%m/%Y %H:%M")
df.set_index("Data", inplace=True)
train, test = train_test_split(df, test_size=0.2, shuffle=False)


train_env = DummyVecEnv([lambda: TradingEnv(train)])
test_env = DummyVecEnv([lambda: TradingEnv(test)])
model = A2C("MlpPolicy", train_env, verbose=1)
model.learn(total_timesteps=16000)
obs = test_env.reset()
done = False
fig = plt.figure(figsize=(15, 6))
for i in range(3900):
    action, _state = model.predict(obs)
    obs, reward, done, info = test_env.step(action)
    test_env.render(mode="human")
    if done:
        break


while True:
    # do something
    if keyboard.is_pressed("q"):
        print("q pressed, ending loop")
        break
