# prbench/ClutteredRetrieval2D-o25-v0
![random action GIF](assets/random_action_gifs/ClutteredRetrieval2D-o25.gif)

### Description
A 2D environment where the goal is to "pick up" (suction) a target block.

The target block may be initially obstructed. In this environment, there are always 25 obstacle blocks.

The robot has a movable circular base and a retractable arm with a rectangular vacuum end effector. Objects can be grasped and ungrasped when the end effector makes contact.

### Initial State Distribution
![initial state GIF](assets/initial_state_gifs/ClutteredRetrieval2D-o25.gif)

### Example Demonstration
![demo GIF](assets/demo_gifs/ClutteredRetrieval2D-o25/ClutteredRetrieval2D-o25_seed4_1752266253.gif)

### Observation Space
The entries of an array in this Box space correspond to the following object features:
| **Index** | **Object** | **Feature** |
| --- | --- | --- |
| 0 | robot | x |
| 1 | robot | y |
| 2 | robot | theta |
| 3 | robot | base_radius |
| 4 | robot | arm_joint |
| 5 | robot | arm_length |
| 6 | robot | vacuum |
| 7 | robot | gripper_height |
| 8 | robot | gripper_width |
| 9 | target_block | x |
| 10 | target_block | y |
| 11 | target_block | theta |
| 12 | target_block | static |
| 13 | target_block | color_r |
| 14 | target_block | color_g |
| 15 | target_block | color_b |
| 16 | target_block | z_order |
| 17 | target_block | width |
| 18 | target_block | height |
| 19 | obstruction0 | x |
| 20 | obstruction0 | y |
| 21 | obstruction0 | theta |
| 22 | obstruction0 | static |
| 23 | obstruction0 | color_r |
| 24 | obstruction0 | color_g |
| 25 | obstruction0 | color_b |
| 26 | obstruction0 | z_order |
| 27 | obstruction0 | width |
| 28 | obstruction0 | height |
| 29 | obstruction1 | x |
| 30 | obstruction1 | y |
| 31 | obstruction1 | theta |
| 32 | obstruction1 | static |
| 33 | obstruction1 | color_r |
| 34 | obstruction1 | color_g |
| 35 | obstruction1 | color_b |
| 36 | obstruction1 | z_order |
| 37 | obstruction1 | width |
| 38 | obstruction1 | height |
| 39 | obstruction10 | x |
| 40 | obstruction10 | y |
| 41 | obstruction10 | theta |
| 42 | obstruction10 | static |
| 43 | obstruction10 | color_r |
| 44 | obstruction10 | color_g |
| 45 | obstruction10 | color_b |
| 46 | obstruction10 | z_order |
| 47 | obstruction10 | width |
| 48 | obstruction10 | height |
| 49 | obstruction11 | x |
| 50 | obstruction11 | y |
| 51 | obstruction11 | theta |
| 52 | obstruction11 | static |
| 53 | obstruction11 | color_r |
| 54 | obstruction11 | color_g |
| 55 | obstruction11 | color_b |
| 56 | obstruction11 | z_order |
| 57 | obstruction11 | width |
| 58 | obstruction11 | height |
| 59 | obstruction12 | x |
| 60 | obstruction12 | y |
| 61 | obstruction12 | theta |
| 62 | obstruction12 | static |
| 63 | obstruction12 | color_r |
| 64 | obstruction12 | color_g |
| 65 | obstruction12 | color_b |
| 66 | obstruction12 | z_order |
| 67 | obstruction12 | width |
| 68 | obstruction12 | height |
| 69 | obstruction13 | x |
| 70 | obstruction13 | y |
| 71 | obstruction13 | theta |
| 72 | obstruction13 | static |
| 73 | obstruction13 | color_r |
| 74 | obstruction13 | color_g |
| 75 | obstruction13 | color_b |
| 76 | obstruction13 | z_order |
| 77 | obstruction13 | width |
| 78 | obstruction13 | height |
| 79 | obstruction14 | x |
| 80 | obstruction14 | y |
| 81 | obstruction14 | theta |
| 82 | obstruction14 | static |
| 83 | obstruction14 | color_r |
| 84 | obstruction14 | color_g |
| 85 | obstruction14 | color_b |
| 86 | obstruction14 | z_order |
| 87 | obstruction14 | width |
| 88 | obstruction14 | height |
| 89 | obstruction15 | x |
| 90 | obstruction15 | y |
| 91 | obstruction15 | theta |
| 92 | obstruction15 | static |
| 93 | obstruction15 | color_r |
| 94 | obstruction15 | color_g |
| 95 | obstruction15 | color_b |
| 96 | obstruction15 | z_order |
| 97 | obstruction15 | width |
| 98 | obstruction15 | height |
| 99 | obstruction16 | x |
| 100 | obstruction16 | y |
| 101 | obstruction16 | theta |
| 102 | obstruction16 | static |
| 103 | obstruction16 | color_r |
| 104 | obstruction16 | color_g |
| 105 | obstruction16 | color_b |
| 106 | obstruction16 | z_order |
| 107 | obstruction16 | width |
| 108 | obstruction16 | height |
| 109 | obstruction17 | x |
| 110 | obstruction17 | y |
| 111 | obstruction17 | theta |
| 112 | obstruction17 | static |
| 113 | obstruction17 | color_r |
| 114 | obstruction17 | color_g |
| 115 | obstruction17 | color_b |
| 116 | obstruction17 | z_order |
| 117 | obstruction17 | width |
| 118 | obstruction17 | height |
| 119 | obstruction18 | x |
| 120 | obstruction18 | y |
| 121 | obstruction18 | theta |
| 122 | obstruction18 | static |
| 123 | obstruction18 | color_r |
| 124 | obstruction18 | color_g |
| 125 | obstruction18 | color_b |
| 126 | obstruction18 | z_order |
| 127 | obstruction18 | width |
| 128 | obstruction18 | height |
| 129 | obstruction19 | x |
| 130 | obstruction19 | y |
| 131 | obstruction19 | theta |
| 132 | obstruction19 | static |
| 133 | obstruction19 | color_r |
| 134 | obstruction19 | color_g |
| 135 | obstruction19 | color_b |
| 136 | obstruction19 | z_order |
| 137 | obstruction19 | width |
| 138 | obstruction19 | height |
| 139 | obstruction2 | x |
| 140 | obstruction2 | y |
| 141 | obstruction2 | theta |
| 142 | obstruction2 | static |
| 143 | obstruction2 | color_r |
| 144 | obstruction2 | color_g |
| 145 | obstruction2 | color_b |
| 146 | obstruction2 | z_order |
| 147 | obstruction2 | width |
| 148 | obstruction2 | height |
| 149 | obstruction20 | x |
| 150 | obstruction20 | y |
| 151 | obstruction20 | theta |
| 152 | obstruction20 | static |
| 153 | obstruction20 | color_r |
| 154 | obstruction20 | color_g |
| 155 | obstruction20 | color_b |
| 156 | obstruction20 | z_order |
| 157 | obstruction20 | width |
| 158 | obstruction20 | height |
| 159 | obstruction21 | x |
| 160 | obstruction21 | y |
| 161 | obstruction21 | theta |
| 162 | obstruction21 | static |
| 163 | obstruction21 | color_r |
| 164 | obstruction21 | color_g |
| 165 | obstruction21 | color_b |
| 166 | obstruction21 | z_order |
| 167 | obstruction21 | width |
| 168 | obstruction21 | height |
| 169 | obstruction22 | x |
| 170 | obstruction22 | y |
| 171 | obstruction22 | theta |
| 172 | obstruction22 | static |
| 173 | obstruction22 | color_r |
| 174 | obstruction22 | color_g |
| 175 | obstruction22 | color_b |
| 176 | obstruction22 | z_order |
| 177 | obstruction22 | width |
| 178 | obstruction22 | height |
| 179 | obstruction23 | x |
| 180 | obstruction23 | y |
| 181 | obstruction23 | theta |
| 182 | obstruction23 | static |
| 183 | obstruction23 | color_r |
| 184 | obstruction23 | color_g |
| 185 | obstruction23 | color_b |
| 186 | obstruction23 | z_order |
| 187 | obstruction23 | width |
| 188 | obstruction23 | height |
| 189 | obstruction24 | x |
| 190 | obstruction24 | y |
| 191 | obstruction24 | theta |
| 192 | obstruction24 | static |
| 193 | obstruction24 | color_r |
| 194 | obstruction24 | color_g |
| 195 | obstruction24 | color_b |
| 196 | obstruction24 | z_order |
| 197 | obstruction24 | width |
| 198 | obstruction24 | height |
| 199 | obstruction3 | x |
| 200 | obstruction3 | y |
| 201 | obstruction3 | theta |
| 202 | obstruction3 | static |
| 203 | obstruction3 | color_r |
| 204 | obstruction3 | color_g |
| 205 | obstruction3 | color_b |
| 206 | obstruction3 | z_order |
| 207 | obstruction3 | width |
| 208 | obstruction3 | height |
| 209 | obstruction4 | x |
| 210 | obstruction4 | y |
| 211 | obstruction4 | theta |
| 212 | obstruction4 | static |
| 213 | obstruction4 | color_r |
| 214 | obstruction4 | color_g |
| 215 | obstruction4 | color_b |
| 216 | obstruction4 | z_order |
| 217 | obstruction4 | width |
| 218 | obstruction4 | height |
| 219 | obstruction5 | x |
| 220 | obstruction5 | y |
| 221 | obstruction5 | theta |
| 222 | obstruction5 | static |
| 223 | obstruction5 | color_r |
| 224 | obstruction5 | color_g |
| 225 | obstruction5 | color_b |
| 226 | obstruction5 | z_order |
| 227 | obstruction5 | width |
| 228 | obstruction5 | height |
| 229 | obstruction6 | x |
| 230 | obstruction6 | y |
| 231 | obstruction6 | theta |
| 232 | obstruction6 | static |
| 233 | obstruction6 | color_r |
| 234 | obstruction6 | color_g |
| 235 | obstruction6 | color_b |
| 236 | obstruction6 | z_order |
| 237 | obstruction6 | width |
| 238 | obstruction6 | height |
| 239 | obstruction7 | x |
| 240 | obstruction7 | y |
| 241 | obstruction7 | theta |
| 242 | obstruction7 | static |
| 243 | obstruction7 | color_r |
| 244 | obstruction7 | color_g |
| 245 | obstruction7 | color_b |
| 246 | obstruction7 | z_order |
| 247 | obstruction7 | width |
| 248 | obstruction7 | height |
| 249 | obstruction8 | x |
| 250 | obstruction8 | y |
| 251 | obstruction8 | theta |
| 252 | obstruction8 | static |
| 253 | obstruction8 | color_r |
| 254 | obstruction8 | color_g |
| 255 | obstruction8 | color_b |
| 256 | obstruction8 | z_order |
| 257 | obstruction8 | width |
| 258 | obstruction8 | height |
| 259 | obstruction9 | x |
| 260 | obstruction9 | y |
| 261 | obstruction9 | theta |
| 262 | obstruction9 | static |
| 263 | obstruction9 | color_r |
| 264 | obstruction9 | color_g |
| 265 | obstruction9 | color_b |
| 266 | obstruction9 | z_order |
| 267 | obstruction9 | width |
| 268 | obstruction9 | height |


### Action Space
The entries of an array in this Box space correspond to the following action features:
| **Index** | **Feature** | **Description** | **Min** | **Max** |
| --- | --- | --- | --- | --- |
| 0 | dx | Change in robot x position (positive is right) | -0.050 | 0.050 |
| 1 | dy | Change in robot y position (positive is up) | -0.050 | 0.050 |
| 2 | dtheta | Change in robot angle in radians (positive is ccw) | -0.196 | 0.196 |
| 3 | darm | Change in robot arm length (positive is out) | -0.100 | 0.100 |
| 4 | vac | Directly sets the vacuum (0.0 is off, 1.0 is on) | 0.000 | 1.000 |


### Rewards
A penalty of -1.0 is given at every time step until termination, which occurs when the target block is held.


### References
Similar environments have been considered by many others, especially in the task and motion planning literature, e.g., "Combined Task and Motion Planning Through an Extensible Planner-Independent Interface Layer" (Srivastava et al., ICRA 2014).
