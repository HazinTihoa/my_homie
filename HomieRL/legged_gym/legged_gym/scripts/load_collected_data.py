import numpy as np
data = np.load('/home/tzh/my_homie/HomieRL/legged_gym/logs/collected_data_play/homie_play_data_20250610-211530.npz')
print(data.files)
print(data['root_states'].shape)
# 打印第0帧第0个机器人的 root_states
print(data['root_states'][0, 0])
print(data['root_states'][10])