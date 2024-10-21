# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.lines import Line2D

# # Define the extrinsic matrix cTw
# cTw = np.array([
#     [0.707, 0.707, 0, -3],
#     [-0.707, 0.707, 0, -0.5],
#     [0, 0, 1, 3]
# ])

# # Define the intrinsic matrix K
# K = np.array([
#     [100, 0, 200],
#     [0, -100, 200],
#     [0, 0, 1]
# ])

# # Define the vertices of the cube in world coordinates
# vertices = np.array([
#     [-0.5, -0.5, -0.5, 1],  # Vertex 0
#     [0.5, -0.5, -0.5, 1],   # Vertex 1
#     [0.5, 0.5, -0.5, 1],    # Vertex 2
#     [-0.5, 0.5, -0.5, 1],   # Vertex 3
#     [-0.5, -0.5, 0.5, 1],   # Vertex 4
#     [0.5, -0.5, 0.5, 1],    # Vertex 5
#     [0.5, 0.5, 0.5, 1],     # Vertex 6
#     [-0.5, 0.5, 0.5, 1],     # Vertex 7
#     [1, -0.5, 0, 1],
#     [1, 0.5, 0, 1]
# ])

# vertices[:, :3] *= 2

# # Project the vertices to the image plane
# projected_vertices = []
# for vertex in vertices:
#     P_c = np.dot(cTw, vertex)  # Transform to camera coordinates
#     p = np.dot(K, P_c)  # Project to image plane
#     p /= p[2]  # Normalize by the third coordinate
#     projected_vertices.append(p[:2])  # Append the 2D coordinates

# projected_vertices = np.array(projected_vertices)

# # Plot the projected vertices and draw lines between them
# fig, ax = plt.subplots()
# ax.set_xlim(-50, 350)
# ax.set_ylim(0, 350)

# # Define the edges of the cube
# edges = [
#     (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
#     (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
#     (0, 4), (1, 5), (2, 6), (3, 7),   # Vertical edges
#     (1, 8), (5, 8), (2, 9), (6, 9), (8,9)
# ]

# # Draw the edges
# for edge in edges:
#     line = Line2D(
#         [projected_vertices[edge[0]][0], projected_vertices[edge[1]][0]],
#         [projected_vertices[edge[0]][1], projected_vertices[edge[1]][1]],
#         linewidth=1, color='blue'
#     )
#     ax.add_line(line)

# plt.gca().invert_yaxis()  # Invert y-axis to match image coordinates
# plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.lines import Line2D

# #importing extrinsic Matrices T and K

# cTw = np.array([
#     [0.707, 0.707, 0, -3],
#     [-0.707, 0.707, 0, -0.5],
#     [0, 0, 1, 3]
# ])

# K = np.array([
#     [100, 0, 200],
#     [0, -100, 200],
#     [0, 0, 1]
# ])

# def main():

#     #Vertices of the Object
#     vertices = np.array([
#         [-0.5, -0.5, -0.5],  
#         [0.5, -0.5, -0.5],   
#         [0.5, 0.5, -0.5],    
#         [-0.5, 0.5, -0.5],   
#         [-0.5, -0.5, 0.5],   
#         [0.5, -0.5, 0.5],    
#         [0.5, 0.5, 0.5],     
#         [-0.5, 0.5, 0.5],    
#         [1, -0.5, 0],        
#         [1, 0.5, 0]          
#     ])
#     ones = np.ones((vertices.shape[0], 1))
#     vertices = np.hstack((vertices, ones))

#     #find the camera matrix using extrinsic matrices
#     camera_matrix = np.dot(K,cTw)

#     #put vertices into imaging pipeline
#     points2d = []
#     for i in vertices:
#         point = np.dot(camera_matrix, i) #point in homogeneous coordinates
#         point = point / point[2] #divide by 3rd element
#         points2d.append(point[:2]) #only take the first two elements for the 2d point

#     points2d = np.array(points2d)
#     #plotting and creating the figure
#     fig, ax = plt.subplots()
    
#     Edges = [
#     (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
#     (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
#     (0, 4), (1, 5), (2, 6), (3, 7),   # Vertical edges
#     (1, 8), (5, 8), (2, 9), (6, 9), (8,9)
#     ]

#     for edge in Edges:
#         line = Line2D(
#             [points2d[edge[0]][0], points2d[edge[1]][0]],
#             [points2d[edge[0]][1], points2d[edge[1]][1]]
#         )
#         ax.add_line(line)

#     #must invert the y axis to match convention

#     ax.set_xlim(min(points2d[:, 0]) - 1, max(points2d[:, 0]) + 1)
#     ax.set_ylim(min(points2d[:, 1]) - 1, max(points2d[:, 1]) + 1)
#     ax.invert_yaxis()
#     plt.show()


    
# if __name__ == "__main__": 
#     main()
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Define your set of vertices
vertices = np.array([
    [0, 0],
    [1, 1],
    [2, 0],
    [3, 1]
])

# Create a figure and axes
fig, ax = plt.subplots()

# Define the edges between vertices
edges = [(0, 1), (1, 2), (2, 3)]

# Add Line2D objects to draw lines between vertices
for edge in edges:
    line = Line2D(
        [vertices[edge[0]][0], vertices[edge[1]][0]],
        [vertices[edge[0]][1], vertices[edge[1]][1]],
        linewidth=2, color='blue'
    )
    ax.add_line(line)



for i, point in enumerate(vertices):
    ax.text(point[0], point[1], str(i), fontsize=12, ha='right')
# Set the limits and show the plot
ax.set_xlim(min(vertices[:, 0]) - 1, max(vertices[:, 0]) + 1)
ax.set_ylim(min(vertices[:, 1]) - 1, max(vertices[:, 1]) + 1)
plt.show()
