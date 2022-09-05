#!/usr/bin/python3
import time
import sys


def func2():
    print("Entering the second function")
    if 'exception' in sys.argv:
        raise Exception("Some interesting exception")
    print("Exiting the second function")


def func1():
    print("Entering the first function")
    func2()
    print("Exiting the second function")


if __name__ == "__main__":
    print(f"Running with args: {sys.argv}")
    time.sleep(2)

    func1()
