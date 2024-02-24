import time
from datetime import datetime
from threading import Thread

# Placeholder for the myTS class with the plot_all_quicklook method
class myTS:
    def __init__(self):
        # Initialize any necessary parameters for myTS
        pass

    def plot_all_quicklook(self, start_date, interval, fig_dir):
        # Simulate plot generation
        print(f"Generating plots for {start_date} in {fig_dir}")

def generate_plots(start_date, end_date):
    current_date = start_date
    while current_date <= end_date:
        year = current_date.year
        month = current_date.month
        savedir = f'/data/QLP/{year}{month:02d}00/Masters/'
        
        # Instantiate myTS anew for each plot generation cycle
        plotter = myTS()
        plotter.plot_all_quicklook(datetime(year, month, 1), interval='month', fig_dir=savedir)
        
        if month == 12:
            current_date = datetime(year + 1, 1, 1)
        else:
            current_date = datetime(year, month + 1, 1)

def schedule_task(initial_delay, interval, start_date, end_date):
    """
    Schedules the plot generation task to run after an initial delay and then at specified intervals,
    allowing for different arguments for each task.

    :param initial_delay: Initial delay in seconds before the task is first executed.
    :param interval: Interval in minutes between task executions.
    :param start_date: Start date for the plot generation.
    :param end_date: End date for the plot generation.
    """
    time.sleep(initial_delay)  # Initial delay before first execution
    while True:
        start_time = time.time()
        generate_plots(start_date, end_date)
        end_time = time.time()
        execution_time = end_time - start_time
        # Calculate sleep time by converting interval from minutes to seconds and subtracting execution time
        sleep_time = interval * 60 - execution_time
        if sleep_time > 0:
            time.sleep(sleep_time)

if __name__ == "__main__":
    # Define different arguments for each task
    tasks = [
        {"initial_delay": 10, "interval": 30, "start_date": datetime(2024, 1, 1), "end_date": datetime(2024, 2, 28)},
        {"initial_delay": 30, "interval": 47, "start_date": datetime(2024, 3, 1), "end_date": datetime(2024, 4, 30)},
        {"initial_delay": 60, "interval": 137, "start_date": datetime(2024, 5, 1), "end_date": datetime(2024, 6, 30)},
    ]

    threads = []
    for task in tasks:
        thread = Thread(target=schedule_task, args=(task["initial_delay"], task["interval"], task["start_date"], task["end_date"]))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()  # Wait for all threads to complete
