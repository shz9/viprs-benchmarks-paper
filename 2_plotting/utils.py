import re
import numpy as np


def extract_performance_statistics(time_file):
    """
    Extract relevant performance statistics from the output of
    /usr/bin/time -v command.

    Primarily, this function, returns the following metrics:
    - Maximum resident set size
    - Wall clock time
    - CPU utilization.

    :param time_file: A path to the file containing the output of /usr/bin/time -v
    :return: A dictionary of parsed performance metrics
    """

    with open(time_file, 'r') as file:
        data = file.read()

    # Extract wall-clock time
    wall_clock_time = re.search(r'Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*(\d*):(\d*\.\d*)', data)
    if wall_clock_time:
        minutes = int(wall_clock_time.group(1))
        seconds = float(wall_clock_time.group(2))
        wall_clock_time = minutes + seconds / 60  # Convert to minutes
    else:
        wall_clock_time = None

    # Extract CPU utilization
    cpu_utilization = re.search(r'Percent of CPU this job got:\s*(\d*)%', data)
    if cpu_utilization:
        cpu_utilization = int(cpu_utilization.group(1))
    else:
        cpu_utilization = None

    # Extract memory utilization
    memory_utilization = re.search(r'Maximum resident set size \(kbytes\):\s*(\d*)', data)
    if memory_utilization:
        memory_utilization = int(memory_utilization.group(1)) / 1024**2  # Convert from KB to GB
    else:
        memory_utilization = None

    return {
        'Wallclock_Time': wall_clock_time,
        'CPU_Util': cpu_utilization,
        'Peak_Memory_GB': memory_utilization
    }


def add_labels_to_bars(g, rotation=90, fontsize='smaller', units=None, orientation='vertical'):
    """
    This function takes a barplot and adds labels above each bar with its value.
    """

    from seaborn.axisgrid import FacetGrid

    if isinstance(g, FacetGrid):
        axes = g.axes.flatten()
    else:
        axes = [g]

    for ax in axes:

        y_min, y_max = ax.get_ylim()
        x_min, x_max = ax.get_xlim()
        scale = ax.get_yaxis().get_scale()

        for p in ax.patches:

            if scale == 'linear':

                height = p.get_height() - y_min
                width = p.get_width() - x_min

                if orientation == 'vertical':
                    value = height
                    x = p.get_x() + p.get_width() / 2
                    if height > 0.5 * y_max:
                        y = y_min + height / 2
                        on_top = False
                    else:
                        y = y_min + height * 1.05
                        on_top = True
                else:  # horizontal barplot
                    value = width
                    y = p.get_y() + p.get_height() / 2
                    if width > 0.5 * x_max:
                        x = x_min + width / 2
                        on_top = False
                    else:
                        x = x_min + width * 1.05
                        on_top = True

            else:

                height = np.log10(p.get_height()) - np.log10(y_min)
                width = np.log10(p.get_width()) - np.log10(x_min)

                if orientation == 'vertical':
                    value = 10 ** height
                    x = p.get_x() + p.get_width() / 2
                    if height > 0.5 * np.log10(y_max):
                        y = 10 ** (np.log10(y_min) + height / 2)
                        on_top = False
                    else:
                        y = 10 ** (np.log10(y_min) + height * 1.05)
                        on_top = True
                else:  # horizontal barplot
                    value = 10 ** width
                    y = p.get_y() + p.get_height() / 2
                    if width > 0.5 * np.log10(x_max):
                        x = 10 ** (np.log10(x_min) + width / 2)
                        on_top = False
                    else:
                        x = 10 ** (np.log10(x_min) + width * 1.05)
                        on_top = True

            label = f'{value:.3f}'
            if units:
                label += f' {units}'

            if orientation == 'horizontal':
                ha = 'left' if on_top else 'center'
                va = 'center'
            else:  # orientation is 'vertical'
                ha = 'center'
                va = 'bottom' if on_top else 'center'

            ax.text(x,
                    y,
                    label,
                    color='black',
                    fontsize=fontsize,
                    rotation=rotation,
                    ha=ha,
                    va=va)
