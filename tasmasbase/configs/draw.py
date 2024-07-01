import matplotlib.pyplot as plt
from commons.functions import get_coordinates, get_center
from .params import SAFETY, LOAD, HOMES, CASES, OBSTACLES

# Define colors
OBSTACLE_COLOR = [1, 0.3, 0.3]
CASE_COLOR = [1, 1, 0.3]
HOME_COLOR = [0.3, 1, 0.4]
LOAD_COLOR = [0, 0.5, 1]
SAFETY_COLOR = [0.8, 0.9, 1]
STATE_COLOR = {'A': [204/255, 0, 0], 'B': [0, 51/255, 204/255], 'C': [204/255, 102/255, 0], 'D': [0, 128/255, 43/255]}


def draw(agents, meas):

    # Define figure
    plt.figure(figsize=(5, 4))

    # Fill figure
    plt.fill(*get_coordinates(SAFETY), color=SAFETY_COLOR)
    plt.fill(*get_coordinates(LOAD), color=LOAD_COLOR)
    plt.text(*get_center(LOAD), 'ULP', fontsize=12, horizontalalignment='center')

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


    for name in agents:
        plt.plot(meas[name][0], meas[name][1], marker='o', color=STATE_COLOR[name], linewidth=2, markersize=4, label=name)


    # Limit figure
    plt.xlim([SAFETY[0], SAFETY[1]])
    plt.ylim([SAFETY[2], SAFETY[3]])

    # Label figure
    plt.xlabel(r'$x_{1}$', fontsize=14)
    plt.xticks(fontsize=12)
    plt.ylabel(r'$x_{2}$', fontsize=14)
    plt.yticks(fontsize=12)
    plt.legend(loc=(0.02, 0.5), fontsize="12", ncol=1)

    # Save figure
    plt.savefig('map.svg', bbox_inches='tight', pad_inches=0.1, transparent=True)

    #Show figure
    plt.show()

    print('Exiting program')
