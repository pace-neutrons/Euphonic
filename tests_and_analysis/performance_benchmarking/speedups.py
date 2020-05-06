import json
import argparse
from typing import Dict

def get_file() -> str:
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", action="store", dest="filename", help="The file to speedrank")
    args_parsed = parser.parse_args()
    return args_parsed.filename

def medianValue(benchmark: Dict) -> float:
    return benchmark["stats"]["median"]

def calculate_speedups(filename: str) -> Dict[str, Dict[int, float]]:
    data = json.load(open(filename))
    data["benchmarks"].sort(key=medianValue) 
    speed_at_threads = {}
    for benchmark in data["benchmarks"]:
        if "use_c" in benchmark["params"] and benchmark["params"]["use_c"] is True:
            func = benchmark["name"].split("[")[0]
            if func not in speed_at_threads:
                speed_at_threads[func] = {}
            speed_at_threads[func][benchmark["params"]["n_threads"]] = benchmark["stats"]["median"]
    speedups = {}
    for func in speed_at_threads:
        speedups[func] = {}
        sequential_speed = speed_at_threads[func][1]
        for n_threads in speed_at_threads[func]:
            speedups[func][n_threads] = sequential_speed / speed_at_threads[func][n_threads]
    return speedups

def write_speedups(filename: str, speedups: Dict[str, Dict[int, float]]):
    data = json.load(open(filename))
    data["speedups"] = speedups
    json.dump(data, open(filename, "w+"), indent=4, sort_keys=True)

if __name__ == "__main__":
    filename: str = get_file()
    speedups: Dict[str, Dict[int, float]] = calculate_speedups(filename)
    write_speedups(filename, speedups)
