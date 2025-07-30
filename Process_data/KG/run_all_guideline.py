import subprocess
import datetime
import sys
import os
import threading
import queue

def run_script(script_name, result_queue):
    """Run the specified Python script and put the result into a queue"""
    print(f"\nStarting to process {script_name}...")
    start_time = datetime.datetime.now()
    
    try:
        # Use subprocess to run the script and capture the output
        result = subprocess.run([sys.executable, script_name], 
                                  capture_output=True, 
                                  text=True, 
                                  encoding='utf-8')
        
        # Print the script's output
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(f"{script_name} error message:", result.stderr)
            
        end_time = datetime.datetime.now()
        duration = end_time - start_time
        print(f"{script_name} processing complete, time taken: {duration}")
        
        # Put the result into the queue
        result_queue.put((script_name, result.returncode == 0, duration))
    except Exception as e:
        print(f"An error occurred while running {script_name}: {str(e)}")
        result_queue.put((script_name, False, None))

def main():
    # Record the start time
    start_time = datetime.datetime.now()
    print(f"Processing start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # List of scripts to run
    scripts = ['NCCN.py', 'CSCO.py', 'ESMO.py']
    # scripts = ['CSCO.py', 'ESMO.py']
    
    # Create a result queue
    result_queue = queue.Queue()
    
    # Create and start threads
    threads = []
    for script in scripts:
        thread = threading.Thread(target=run_script, args=(script, result_queue))
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    # Collect results
    failed_scripts = []
    for _ in range(len(scripts)):
        script_name, success, duration = result_queue.get()
        if not success:
            failed_scripts.append(script_name)
    
    # Record the end time
    end_time = datetime.datetime.now()
    total_duration = end_time - start_time
    print(f"\nAll processing finished at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total time taken: {total_duration}")
    
    # Report failed scripts
    if failed_scripts:
        print("\nThe following scripts failed to run:")
        for script in failed_scripts:
            print(f"- {script}")

if __name__ == "__main__":
    main()