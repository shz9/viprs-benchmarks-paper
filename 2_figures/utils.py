import re


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

