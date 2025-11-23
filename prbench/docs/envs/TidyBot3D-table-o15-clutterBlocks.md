# prbench/TidyBot3D-table-o15-clutterBlocks-v0
![random action GIF](assets/random_action_gifs/TidyBot3D-table-o15-clutterBlocks.gif)

### Description
A 3D mobile manipulation environment using the TidyBot platform.

The robot has a holonomic mobile base with powered casters and a Kinova Gen3 arm.
Scene type: table with 15 objects.

The robot can control:
- Base pose (x, y, theta)
- Arm position (x, y, z)
- Arm orientation (quaternion)
- Gripper position (open/close)

### Initial State Distribution
![initial state GIF](assets/initial_state_gifs/TidyBot3D-table-o15-clutterBlocks.gif)

### Example Demonstration
*(No demonstration GIFs available)*

### Observation Space
The entries of an array in this Box space correspond to the following object features:
| **Index** | **Object** | **Feature** |
| --- | --- | --- |
| 0 | cube1 | x |
| 1 | cube1 | y |
| 2 | cube1 | z |
| 3 | cube1 | qw |
| 4 | cube1 | qx |
| 5 | cube1 | qy |
| 6 | cube1 | qz |
| 7 | cube1 | vx |
| 8 | cube1 | vy |
| 9 | cube1 | vz |
| 10 | cube1 | wx |
| 11 | cube1 | wy |
| 12 | cube1 | wz |
| 13 | cube1 | bb_x |
| 14 | cube1 | bb_y |
| 15 | cube1 | bb_z |
| 16 | cube10 | x |
| 17 | cube10 | y |
| 18 | cube10 | z |
| 19 | cube10 | qw |
| 20 | cube10 | qx |
| 21 | cube10 | qy |
| 22 | cube10 | qz |
| 23 | cube10 | vx |
| 24 | cube10 | vy |
| 25 | cube10 | vz |
| 26 | cube10 | wx |
| 27 | cube10 | wy |
| 28 | cube10 | wz |
| 29 | cube10 | bb_x |
| 30 | cube10 | bb_y |
| 31 | cube10 | bb_z |
| 32 | cube11 | x |
| 33 | cube11 | y |
| 34 | cube11 | z |
| 35 | cube11 | qw |
| 36 | cube11 | qx |
| 37 | cube11 | qy |
| 38 | cube11 | qz |
| 39 | cube11 | vx |
| 40 | cube11 | vy |
| 41 | cube11 | vz |
| 42 | cube11 | wx |
| 43 | cube11 | wy |
| 44 | cube11 | wz |
| 45 | cube11 | bb_x |
| 46 | cube11 | bb_y |
| 47 | cube11 | bb_z |
| 48 | cube12 | x |
| 49 | cube12 | y |
| 50 | cube12 | z |
| 51 | cube12 | qw |
| 52 | cube12 | qx |
| 53 | cube12 | qy |
| 54 | cube12 | qz |
| 55 | cube12 | vx |
| 56 | cube12 | vy |
| 57 | cube12 | vz |
| 58 | cube12 | wx |
| 59 | cube12 | wy |
| 60 | cube12 | wz |
| 61 | cube12 | bb_x |
| 62 | cube12 | bb_y |
| 63 | cube12 | bb_z |
| 64 | cube13 | x |
| 65 | cube13 | y |
| 66 | cube13 | z |
| 67 | cube13 | qw |
| 68 | cube13 | qx |
| 69 | cube13 | qy |
| 70 | cube13 | qz |
| 71 | cube13 | vx |
| 72 | cube13 | vy |
| 73 | cube13 | vz |
| 74 | cube13 | wx |
| 75 | cube13 | wy |
| 76 | cube13 | wz |
| 77 | cube13 | bb_x |
| 78 | cube13 | bb_y |
| 79 | cube13 | bb_z |
| 80 | cube14 | x |
| 81 | cube14 | y |
| 82 | cube14 | z |
| 83 | cube14 | qw |
| 84 | cube14 | qx |
| 85 | cube14 | qy |
| 86 | cube14 | qz |
| 87 | cube14 | vx |
| 88 | cube14 | vy |
| 89 | cube14 | vz |
| 90 | cube14 | wx |
| 91 | cube14 | wy |
| 92 | cube14 | wz |
| 93 | cube14 | bb_x |
| 94 | cube14 | bb_y |
| 95 | cube14 | bb_z |
| 96 | cube15 | x |
| 97 | cube15 | y |
| 98 | cube15 | z |
| 99 | cube15 | qw |
| 100 | cube15 | qx |
| 101 | cube15 | qy |
| 102 | cube15 | qz |
| 103 | cube15 | vx |
| 104 | cube15 | vy |
| 105 | cube15 | vz |
| 106 | cube15 | wx |
| 107 | cube15 | wy |
| 108 | cube15 | wz |
| 109 | cube15 | bb_x |
| 110 | cube15 | bb_y |
| 111 | cube15 | bb_z |
| 112 | cube2 | x |
| 113 | cube2 | y |
| 114 | cube2 | z |
| 115 | cube2 | qw |
| 116 | cube2 | qx |
| 117 | cube2 | qy |
| 118 | cube2 | qz |
| 119 | cube2 | vx |
| 120 | cube2 | vy |
| 121 | cube2 | vz |
| 122 | cube2 | wx |
| 123 | cube2 | wy |
| 124 | cube2 | wz |
| 125 | cube2 | bb_x |
| 126 | cube2 | bb_y |
| 127 | cube2 | bb_z |
| 128 | cube3 | x |
| 129 | cube3 | y |
| 130 | cube3 | z |
| 131 | cube3 | qw |
| 132 | cube3 | qx |
| 133 | cube3 | qy |
| 134 | cube3 | qz |
| 135 | cube3 | vx |
| 136 | cube3 | vy |
| 137 | cube3 | vz |
| 138 | cube3 | wx |
| 139 | cube3 | wy |
| 140 | cube3 | wz |
| 141 | cube3 | bb_x |
| 142 | cube3 | bb_y |
| 143 | cube3 | bb_z |
| 144 | cube4 | x |
| 145 | cube4 | y |
| 146 | cube4 | z |
| 147 | cube4 | qw |
| 148 | cube4 | qx |
| 149 | cube4 | qy |
| 150 | cube4 | qz |
| 151 | cube4 | vx |
| 152 | cube4 | vy |
| 153 | cube4 | vz |
| 154 | cube4 | wx |
| 155 | cube4 | wy |
| 156 | cube4 | wz |
| 157 | cube4 | bb_x |
| 158 | cube4 | bb_y |
| 159 | cube4 | bb_z |
| 160 | cube5 | x |
| 161 | cube5 | y |
| 162 | cube5 | z |
| 163 | cube5 | qw |
| 164 | cube5 | qx |
| 165 | cube5 | qy |
| 166 | cube5 | qz |
| 167 | cube5 | vx |
| 168 | cube5 | vy |
| 169 | cube5 | vz |
| 170 | cube5 | wx |
| 171 | cube5 | wy |
| 172 | cube5 | wz |
| 173 | cube5 | bb_x |
| 174 | cube5 | bb_y |
| 175 | cube5 | bb_z |
| 176 | cube6 | x |
| 177 | cube6 | y |
| 178 | cube6 | z |
| 179 | cube6 | qw |
| 180 | cube6 | qx |
| 181 | cube6 | qy |
| 182 | cube6 | qz |
| 183 | cube6 | vx |
| 184 | cube6 | vy |
| 185 | cube6 | vz |
| 186 | cube6 | wx |
| 187 | cube6 | wy |
| 188 | cube6 | wz |
| 189 | cube6 | bb_x |
| 190 | cube6 | bb_y |
| 191 | cube6 | bb_z |
| 192 | cube7 | x |
| 193 | cube7 | y |
| 194 | cube7 | z |
| 195 | cube7 | qw |
| 196 | cube7 | qx |
| 197 | cube7 | qy |
| 198 | cube7 | qz |
| 199 | cube7 | vx |
| 200 | cube7 | vy |
| 201 | cube7 | vz |
| 202 | cube7 | wx |
| 203 | cube7 | wy |
| 204 | cube7 | wz |
| 205 | cube7 | bb_x |
| 206 | cube7 | bb_y |
| 207 | cube7 | bb_z |
| 208 | cube8 | x |
| 209 | cube8 | y |
| 210 | cube8 | z |
| 211 | cube8 | qw |
| 212 | cube8 | qx |
| 213 | cube8 | qy |
| 214 | cube8 | qz |
| 215 | cube8 | vx |
| 216 | cube8 | vy |
| 217 | cube8 | vz |
| 218 | cube8 | wx |
| 219 | cube8 | wy |
| 220 | cube8 | wz |
| 221 | cube8 | bb_x |
| 222 | cube8 | bb_y |
| 223 | cube8 | bb_z |
| 224 | cube9 | x |
| 225 | cube9 | y |
| 226 | cube9 | z |
| 227 | cube9 | qw |
| 228 | cube9 | qx |
| 229 | cube9 | qy |
| 230 | cube9 | qz |
| 231 | cube9 | vx |
| 232 | cube9 | vy |
| 233 | cube9 | vz |
| 234 | cube9 | wx |
| 235 | cube9 | wy |
| 236 | cube9 | wz |
| 237 | cube9 | bb_x |
| 238 | cube9 | bb_y |
| 239 | cube9 | bb_z |
| 240 | robot | pos_base_x |
| 241 | robot | pos_base_y |
| 242 | robot | pos_base_rot |
| 243 | robot | pos_arm_joint1 |
| 244 | robot | pos_arm_joint2 |
| 245 | robot | pos_arm_joint3 |
| 246 | robot | pos_arm_joint4 |
| 247 | robot | pos_arm_joint5 |
| 248 | robot | pos_arm_joint6 |
| 249 | robot | pos_arm_joint7 |
| 250 | robot | pos_gripper |
| 251 | robot | vel_base_x |
| 252 | robot | vel_base_y |
| 253 | robot | vel_base_rot |
| 254 | robot | vel_arm_joint1 |
| 255 | robot | vel_arm_joint2 |
| 256 | robot | vel_arm_joint3 |
| 257 | robot | vel_arm_joint4 |
| 258 | robot | vel_arm_joint5 |
| 259 | robot | vel_arm_joint6 |
| 260 | robot | vel_arm_joint7 |
| 261 | robot | vel_gripper |
| 262 | table_1 | x |
| 263 | table_1 | y |
| 264 | table_1 | z |
| 265 | table_1 | qw |
| 266 | table_1 | qx |
| 267 | table_1 | qy |
| 268 | table_1 | qz |


### Action Space
Actions: base_pose (3), arm_pos (3), arm_quat (4), gripper_pos (1)

### Rewards
Reward function depends on the specific task:
- Object stacking: Reward for successfully stacking objects
- Drawer/cabinet tasks: Reward for opening/closing and placing objects
- General manipulation: Reward for successful pick-and-place operations

Currently returns a small negative reward (-0.01) per timestep to encourage exploration.


### References
TidyBot++: An Open-Source Holonomic Mobile Manipulator
for Robot Learning
- Jimmy Wu, William Chong, Robert Holmberg, Aaditya Prasad, Yihuai Gao,
  Oussama Khatib, Shuran Song, Szymon Rusinkiewicz, Jeannette Bohg
- Conference on Robot Learning (CoRL), 2024

https://github.com/tidybot2/tidybot2
