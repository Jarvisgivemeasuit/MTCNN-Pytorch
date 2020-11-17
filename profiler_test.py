import time
def profiler(func):
    def wrapper(*args, **kwargs):
        tic = time.time()
        res = func(*args, **kwargs)
        print(f'"{func.__name__}" executed in {time.time() - tic:.7f} s')
        return res

    return wrapper
    
@profiler
def add(a, b):
    if __debug__:
        print('debug mode.')
    return a+b

if __name__ == '__main__':
    add(2, 4)
