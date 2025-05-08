import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_spaced_cnn_3d():
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Colors
    conv_color = '#1f77b4'
    pool_color = '#ff7f0e'
    fc_color = '#2ca02c'
    activ_color = '#d62728'
    flatten_color = '#9467bd'
    
    # Layer configuration
    layers = [
        {'type': 'input', 'name': 'Input\n1×256×256', 'color': 'gray', 'width': 1, 'height': 2, 'depth': 2},
        {'type': 'conv', 'name': 'Conv1\n16×256×256\nk3×3, p1', 'color': conv_color, 'width': 1.2, 'height': 1.8, 'depth': 1.8},
        {'type': 'activ', 'name': 'ReLU', 'color': activ_color, 'width': 0.8, 'height': 0.8, 'depth': 0.8},
        {'type': 'pool', 'name': 'MaxPool1\n16×128×128\ns2', 'color': pool_color, 'width': 1, 'height': 1.6, 'depth': 1.6},
        {'type': 'conv', 'name': 'Conv2\n32×128×128\nk3×3, p1', 'color': conv_color, 'width': 1.4, 'height': 1.4, 'depth': 1.4},
        {'type': 'activ', 'name': 'ReLU', 'color': activ_color, 'width': 0.8, 'height': 0.8, 'depth': 0.8},
        {'type': 'pool', 'name': 'MaxPool2\n32×64×64\ns2', 'color': pool_color, 'width': 1.2, 'height': 1.2, 'depth': 1.2},
        {'type': 'conv', 'name': 'Conv3\n64×64×64\nk3×3, p1', 'color': conv_color, 'width': 1.6, 'height': 1.0, 'depth': 1.0},
        {'type': 'activ', 'name': 'ReLU', 'color': activ_color, 'width': 0.8, 'height': 0.8, 'depth': 0.8},
        {'type': 'pool', 'name': 'MaxPool3\n64×32×32\ns2', 'color': pool_color, 'width': 1.4, 'height': 0.8, 'depth': 0.8},
        {'type': 'flatten', 'name': 'Flatten\n64×16×16', 'color': flatten_color, 'width': 1, 'height': 0.5, 'depth': 3},
        {'type': 'fc', 'name': 'FC1\n128\nDropout(0.3)', 'color': fc_color, 'width': 1.2, 'height': 0.4, 'depth': 2},
        {'type': 'fc', 'name': f'FC2\n{num_classes} classes', 'color': fc_color, 'width': 1, 'height': 0.3, 'depth': 1.5}
    ]
    
    # Position layers with proper spacing
    x_positions = np.cumsum([0] + [layer['width']*1.2 for layer in layers[:-1]])
    
    # Plot each layer
    for i, (layer, x) in enumerate(zip(layers, x_positions)):
        # Create layer shape
        ax.bar3d(x, 0, 0, 
                layer['width'], layer['height'], layer['depth'],
                color=layer['color'], alpha=0.8, edgecolor='black')
        
        # Add layer label
        ax.text(x + layer['width']/2, layer['height']/2, layer['depth'] + 0.2,
               layer['name'], ha='center', va='bottom', fontsize=9, color='black')
        
        # Add connections between layers
        if i > 0:
            prev_layer = layers[i-1]
            ax.plot3D([x, x - layer['width']*0.2],
                     [prev_layer['height']/2, layer['height']/2],
                     [prev_layer['depth']/2, layer['depth']/2],
                     'gray', alpha=0.5, linewidth=1)
    
    # Configure view
    ax.set_xlabel('Network Depth')
    ax.set_ylabel('Feature Map Size')
    ax.set_zlabel('Channels/Neurons')
    ax.set_title('3D CNN Architecture Visualization', pad=20)
    ax.view_init(elev=25, azim=-45)
    ax.dist = 10  # Adjust camera distance
    
    # Remove tick labels for cleaner look
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    
    plt.tight_layout()
    plt.show()

# Set your number of classes
num_classes = 10  # Change as needed
plot_spaced_cnn_3d()