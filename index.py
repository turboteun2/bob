import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import json
import random  # Voeg deze import toe

# Stap 1: Stickman tekenen met voet op de grondlijn
def draw_stickman(ax, x_offset=0, y_offset=0, leg_angle=0, arm_angle=0):
    # Hoofd
    hoofd = plt.Circle((x_offset + 5, y_offset + 8), 1, fill=False)
    ax.add_patch(hoofd)
    
    # Lichaam
    ax.plot([x_offset + 5, x_offset + 5], [y_offset + 7, y_offset + 4], 'k-')
    
    # Benen - bewegingen laten oscilleren om te simuleren dat de stickman loopt
    ax.plot([x_offset + 5, x_offset + 5 - np.cos(np.radians(leg_angle))], [y_offset + 4, y_offset + 2], 'k-')
    ax.plot([x_offset + 5, x_offset + 5 + np.cos(np.radians(leg_angle))], [y_offset + 4, y_offset + 2], 'k-')
    
    # Armen (nu ook bewegend op basis van beenbeweging)
    arm_length = 2  # Lengte van de arm
    # Beweging armen wordt aan de tegenovergestelde kant van de benen berekend
    ax.plot([x_offset + 5 - arm_length, x_offset + 5 - arm_length * np.cos(np.radians(arm_angle))], 
            [y_offset + 6, y_offset + 6 - arm_length * np.sin(np.radians(arm_angle))], 'k-')  # Links arm
    ax.plot([x_offset + 5 + arm_length, x_offset + 5 + arm_length * np.cos(np.radians(arm_angle))], 
            [y_offset + 6, y_offset + 6 - arm_length * np.sin(np.radians(arm_angle))], 'k-')  # Rechts arm

# Stap 2: Genereer random animatie data en sla op
def random_animation_data(num_frames):
    data = []
    for _ in range(num_frames):
        leg_angle = random.uniform(0, 30)  # Genereer willekeurige hoeken voor beenbeweging
        arm_angle = -leg_angle  # Armen gaan tegenovergesteld aan benen
        data.append((leg_angle, arm_angle))
    return data

animation_data = random_animation_data(100)
with open('stickman_animation.json', 'w') as file:
    json.dump(animation_data, file)

# Stap 3: Lees data voor animatie
with open("stickman_animation.json", "r") as file:
    animation_data = json.load(file)

# Stap 4: Animeren van stickman, zoals als hij aan het lopen is
fig, ax = plt.subplots()
ax.set_aspect('equal')

def animate(frame):
    ax.cla()  # Clear het plot zonder limieten
    ax.set_xlim(0, 10)  # Zet een aslimiet
    ax.set_ylim(0, 10)   # Zet een aslimiet
    frame_data = animation_data[frame]  # Haal uit de JSON
    x_offset = 0.1 * frame  # Verplaats stickman langs de x-as
    y_offset = 0  # Grondlijn blijft op dezelfde hoogte
    leg_angle, arm_angle = frame_data  # Winkel van het been voor lopen
    draw_stickman(ax, x_offset=x_offset, y_offset=y_offset, leg_angle=leg_angle, arm_angle=arm_angle)

ani = FuncAnimation(fig, animate, frames=len(animation_data), interval=200)
plt.show()
