import matplotlib.pyplot as plt
from utils.functions import get_coordinates, get_center
from config import SAFETY, PICKUP, HOMES, CASES, OBSTACLES
import os
import pickle

# Define colors
OBSTACLE_COLOR = [255/255, 204/255, 204/255]
CASE_COLOR = [229/255, 245/255, 255/255]
HOME_COLOR = [229/255, 255/255, 229/255]
PICKUP_COLOR = [224/255, 224/255, 235/255]
SAFETY_COLOR = [255/255, 255/255, 229/255]
STATE_COLOR = {0: [204/255, 0, 0], 1: [0, 51/255, 204/255], 2: [204/255, 102/255, 0], 3: [0, 128/255, 43/255]}

names = ['A', 'B', 'C', 'D']

def draw(measures):

    # Define figure
    plt.figure(figsize=(5, 4))

    # Fill figure
    plt.fill(*get_coordinates(SAFETY), color=SAFETY_COLOR)
    plt.fill(*get_coordinates(PICKUP), color=PICKUP_COLOR)
    plt.text(*get_center(PICKUP), 'PICKUP', fontsize=12, horizontalalignment='center')

    for name, region in CASES.items():
        plt.fill(*get_coordinates(region), color=CASE_COLOR)
        plt.plot(*get_coordinates(region), linewidth=2, color=(0, 0, 0))
        plt.text(*get_center(region), name, fontsize=12, horizontalalignment='center')
    for name, region in HOMES.items():
        plt.fill(*get_coordinates(region), color=HOME_COLOR)
        plt.plot(*get_coordinates(region), linewidth=2, color=(0, 0, 0))
        plt.text(*get_center(region), name, fontsize=12, horizontalalignment='center')
    for name, region in OBSTACLES.items():
        plt.fill(*get_coordinates(region), color=OBSTACLE_COLOR)
        plt.text(*get_center(region), name, fontsize=12, horizontalalignment='center')


    for name, measure in measures.items():
        plt.plot(measure[0], measure[1], marker='o', color=STATE_COLOR[name], linewidth=2, markersize=4, label=names[name])


    # Limit figure
    plt.xlim([SAFETY[0], SAFETY[1]])
    plt.ylim([SAFETY[2], SAFETY[3]])

    # Label figure
    plt.xlabel(r'$x_{1}$', fontsize=14)
    plt.xticks(fontsize=12)
    plt.ylabel(r'$x_{2}$', fontsize=14)
    plt.yticks(fontsize=12)
    plt.legend(loc=(0.8, 0.6), fontsize="12", ncol=1)

    # Save figure
    if not os.path.exists('figures'):
        os.makedirs('figures')

    plt.savefig(os.path.join('figures', 'map.svg'), bbox_inches='tight', pad_inches=0.1, transparent=True)

    #Show figure
    plt.show()

    print('Exiting program')


if __name__ == "__main__":

    with open('data.pkl', 'rb') as f:
        measures = pickle.load(f)
        draw(measures)