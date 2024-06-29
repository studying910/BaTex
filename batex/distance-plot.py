import matplotlib.pyplot as plt
import numpy as np

n = 1050
lw = '3'
step = np.arange(50, n, 50)

# 1: dot, 2: cos, 3: l2

y1_gta5 = [0.682, 0.716, 0.716, 0.792, 0.812, 0.805, 0.802, 0.800, 0.791, 0.802, 0.794, 0.798, 0.817, 0.800, 0.811,
           0.801, 0.799, 0.768, 0.779, 0.803]
y2_gta5 = [0.680, 0.700, 0.710, 0.795, 0.805, 0.787, 0.797, 0.807, 0.782, 0.792, 0.795, 0.785, 0.807, 0.788, 0.792,
           0.769, 0.766, 0.769, 0.758, 0.782]
y3_gta5 = [0.679, 0.677, 0.675, 0.678, 0.681, 0.692, 0.711, 0.723, 0.739, 0.747, 0.745, 0.757, 0.736, 0.753, 0.756,
           0.762, 0.764, 0.767, 0.761, 0.766]

y1_anders = [0.703, 0.731, 0.746, 0.745, 0.761, 0.768, 0.773, 0.768, 0.763, 0.770, 0.752, 0.777, 0.789, 0.764, 0.782,
             0.758, 0.756, 0.771, 0.779, 0.755]
y2_anders = [0.688, 0.699, 0.707, 0.698, 0.729, 0.706, 0.739, 0.753, 0.748, 0.740, 0.741, 0.755, 0.730, 0.760, 0.741,
             0.741, 0.742, 0.761, 0.727, 0.749]
y3_anders = [0.703, 0.704, 0.703, 0.706, 0.701, 0.707, 0.709, 0.718, 0.718, 0.733, 0.746, 0.764, 0.753, 0.750, 0.750,
             0.745, 0.739, 0.762, 0.764, 0.749]

fig, ax = plt.subplots(1, 2, constrained_layout=True, sharey="all", figsize=(8, 4))
# plt.subplot(1, 2, 1)
ax[0].plot(step, y1_gta5, linewidth=lw, label="Dot product", color="#0000FF")  # , color = '#1f77b4')
ax[0].plot(step, y2_gta5, "--", linewidth=lw, label="Cosine sinilarity", color="#FF0000")
ax[0].plot(step, y3_gta5, ":", linewidth="4", label="L2", color="#000000")

ax[0].legend(fontsize=12, loc='lower right', ncol=1)
ax[0].set_xlim([50, 1000])
ax[0].set_ylim([0.0, 1.0])
# fig = plt.figure()
ax[0].set_title("Gta5-artwork", fontsize=18)
ax[0].set_xlabel("Training Steps", fontsize=18)
ax[0].set_ylabel("Text-image Alignment Score", fontsize=18)
ax[0].set_xticks([50, 200, 400, 600, 800, 1000])
ax[0].set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax[0].set_xticklabels([50, 200, 400, 600, 800, 1000], fontsize=12)
ax[0].set_yticklabels([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=12)

# plt.subplot(1, 2, 2)
ax[1].plot(step, y1_anders, linewidth=lw, label="Dot product", color="#0000FF")  # , color = '#1f77b4')
ax[1].plot(step, y2_anders, "--", linewidth=lw, label="Cosine sinilarity", color="#FF0000")
ax[1].plot(step, y3_anders, ":", linewidth="4", label="L2", color="#000000")

ax[1].legend(fontsize=12, loc='lower right', ncol=1)
ax[1].set_xlim([50, 1000])
ax[1].set_ylim([0.0, 1.0])
# fig = plt.figure()
ax[1].set_title("Anders-zorn", fontsize=18)
ax[1].set_xlabel("Training Steps", fontsize=18)
# ax[1].set_ylabel("Text-image Alignment Score", fontsize=15)
ax[1].set_xticks([50, 200, 400, 600, 800, 1000])
ax[1].set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax[1].set_xticklabels([50, 200, 400, 600, 800, 1000], fontsize=12)
ax[1].set_yticklabels([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=12)

plt.show()
