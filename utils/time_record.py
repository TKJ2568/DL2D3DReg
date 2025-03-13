"""
用于记录某个函数的运行时间
"""
import time
from functools import wraps

def timer_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()  # 记录开始时间
        result = func(*args, **kwargs)    # 执行被装饰的函数
        end_time = time.perf_counter()    # 记录结束时间
        elapsed_time = end_time - start_time  # 计算执行时间
        print(f"Function {func.__name__} executed in {elapsed_time:.6f} seconds")
        return result  # 返回被装饰函数的执行结果
    return wrapper

def timer_decorator_with_info(info_message):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()  # 记录开始时间
            print(f"{info_message}")
            result = func(*args, **kwargs)    # 执行被装饰的函数
            end_time = time.perf_counter()    # 记录结束时间
            elapsed_time = end_time - start_time  # 计算执行时间
            print(f"Function {func.__name__} executed in {elapsed_time:.6f} seconds")
            return result  # 返回被装饰函数的执行结果
        return wrapper
    return decorator