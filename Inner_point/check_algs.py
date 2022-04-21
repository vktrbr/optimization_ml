import ctypes
from time import time_ns


def factorial(_n: int) -> int:
    fact = 1
    for i in range(1, _n + 1):
        fact *= i
    return fact


go_pack = ctypes.CDLL('./algorithms_go.dll')

n = 1000
start_time = time_ns()
print(go_pack.factorial(n))
print(time_ns() - start_time)

start_time = time_ns()
print(factorial(n))
print(time_ns() - start_time)
