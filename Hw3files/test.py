import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

#cTw
cTw = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 3]
])

K = np.array([
    [100, 0, 200],
    [0, -100, 200],
    [0, 0, 1]
])
print(K)
vertices = np.array([[1, 1, 0, 1],
                     [1, -1, 0, 1],
                     [-1, -1, 0, 1],
                     [-1, 1, 0, 1],
                     [1, 1, 2, 1],
                     [1, -1, 2, 1],
                     [-1, -1, 2, 1],
                     [-1, 1, 2, 1],
                     [0, 0, 3, 1]
                     ]) 

print(vertices)

projected_vertices = []
for vertex in vertices:
    print(vertex)
    
    vertex_homogeneous = np.array([vertex[0], vertex[1], vertex[2], 1])
    
    P_c = np.dot(cTw, vertex_homogeneous)
    print(P_c)
    
    p = np.dot(K, P_c)
    p = p/p[2]  
    projected_vertices.append(p[:2])

edges = [
    [0, 1], [1, 2], [2, 3], [3, 0],  #bottom
    [4, 5], [5, 6], [6, 7], [7, 4],  #top
    [0, 4], [1, 5], [2, 6], [3, 7],  #vert
    [4, 8], [5, 8], [6, 8], [7, 8]   #roof
]

projected_vertices = np.array(projected_vertices)

fig, ax = plt.subplots()
# ax.set_xlim(250, 3000)
# ax.set_ylim(500, 4000)
ax.set_xlim(min(projected_vertices[:, 0]) - 10, max(projected_vertices[:, 0]) + 10)
ax.set_ylim(min(projected_vertices[:, 1]) - 10, max(projected_vertices[:, 1]) + 10)

for edge in edges:
    line = Line2D(
        [projected_vertices[edge[0]][0], projected_vertices[edge[1]][0]],
        [projected_vertices[edge[0]][1], projected_vertices[edge[1]][1]],
        linewidth=1, color='blue'
    )
    ax.add_line(line)
 
plt.show()