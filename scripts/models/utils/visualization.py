import imageio
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from skimage.transform import resize
from models.utils import graph
import matplotlib.text as mtext
import numpy as np
import imgviz
# get path with
# $ which ffmpeg
plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
def display_video(video):
    fig = plt.figure(figsize=(3,3))  #Display size specification

    mov = []
    for i in range(len(video)):  #Append videos one by one to mov
        img = plt.imshow(video[i], animated=True)
        plt.axis('off')
        mov.append([img])

    #Animation creation
    anime = animation.ArtistAnimation(fig, mov, interval=50, repeat_delay=1000)

    plt.close()
    return anime

def display_video_lbl(video, lbls):

    lbls = np.repeat(lbls, 48)
    lbls = np.where(lbls == 0, 'no-mov', 'mov')

    fig, ax = plt.subplots(figsize=(3,3))  #Display size specification

    mov = []
    for i in range(len(video)):  #Append videos one by one to mov
        img = plt.imshow(video[i], animated=True)
        plt.axis('off')
        title = mtext.Text(0.15, 0.9, f'{lbls[i]}', transform=ax.transAxes,
                           ha='center', color='r')
        ax.add_artist(title)
        mov.append([img, title])

    #Animation creation
    anime = animation.ArtistAnimation(fig, mov, interval=50, repeat_delay=1000)

    plt.close()
    return anime

# def viz_pose(data):
#     # Data preprocessing
#     data[data[:, :, :, -1] == 0.5] = 0.0
#     data[data[:, :, :, -1] == -0.5] = 0.0

#     C, T, V, M = data.shape

#     # Define the connections in your skeleton
#     coco_inward = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 5),
#                 (12, 6), (9, 7), (7, 5), (10, 8), (8, 6), (5, 0),
#                 (6, 0), (1, 0), (3, 1), (2, 0), (4, 2)]
#     edge = coco_inward

#     # Prepare the figure
#     fig, ax = plt.subplots(figsize=(3,3))
#     ax.grid(True)
#     ax.axis([-1, 1, -1, 1])

#     # This function is called to update the plot for each frame
#     def update_graph(t):
#         ax.clear()  # Clear previous frame
#         ax.grid(True)
#         ax.axis([-1, 1, -1, 1])
#         for m in range(M):
#             for v1, v2 in edge:
#                 ax.plot(data[0, t, [v1, v2], m], -data[1, t, [v1, v2], m], 'b-')
#         return ax

#     # Create the animation
#     ani = animation.FuncAnimation(fig, update_graph, frames=T, interval=50)

#     plt.close()  # Prevents the static plot from showing
#     return ani

def viz_pose(data, face, r_hand, l_hand):
    C, T, V, M = data.shape

    # Prepare the figure
    fig, ax = plt.subplots(figsize=(3,3))
    ax.grid(True)
    ax.axis([-1, 1, -1, 1])

    # This function is called to update the plot for each frame
    def update_graph(t):
        ax.clear()  # Clear previous frame
        ax.grid(True)
        ax.axis([-1, 1, -1, 1])
        for m in range(M):
            for v1, v2 in graph.coco_inward_edges:
                ax.plot(data[0, t, [v1, v2], m], -data[1, t, [v1, v2], m], 'b-')
            for v1, v2 in graph.hand_inward_edges:
                ax.plot(r_hand[0, t, [v1, v2], m], -r_hand[1, t, [v1, v2], m], 'r-')
            for v1, v2 in graph.hand_inward_edges:
                ax.plot(l_hand[0, t, [v1, v2], m], -l_hand[1, t, [v1, v2], m], 'g-')
            for v1, v2 in graph.face_inward_edges:
                ax.plot(face[0, t, [v1, v2], m], -face[1, t, [v1, v2], m], 'c-')
        return ax

    # Create the animation
    ani = animation.FuncAnimation(fig, update_graph, frames=T, interval=50)

    plt.close()  # Prevents the static plot from showing
    return ani


# import numpy as np
# import matplotlib.pyplot as plt

# # Values for alpha and beta
# alpha_values = [0.5, 1, 2, 3, 1, 3]
# beta_values = [0.5, 1, 2, 3, 3, 1]
# colors = ['b', 'g', 'r', 'c', 'm', 'y']

# plt.figure(figsize=(14, 10))

# for i, (alpha, beta) in enumerate(zip(alpha_values, beta_values)):
#     # Generate a beta distribution sample
#     samples = np.random.beta(alpha, beta, size=10000)
    
#     # Plot the histogram of samples
#     plt.subplot(3, 2, i+1)
#     plt.hist(samples, bins=50, color=colors[i], alpha=0.7, density=True)
#     plt.title(f'alpha = {alpha}, beta = {beta}')
#     plt.xlabel('Value')
#     plt.ylabel('Density')

# plt.tight_layout()
# plt.show()

def show_batch_frames(batch, batch_size=4):
    x=[]
    for i in range(batch_size):
        x.append(batch['vid_i'][i].numpy()[0,...].astype(np.uint8))
        x.append(batch['vid_j'][i].numpy()[0,...].astype(np.uint8))
    tiles = imgviz.tile(x, shape=(4, int(batch_size/2)), border=(255,0,0))
    plt.imshow(tiles)
    plt.axis('off')
    return None