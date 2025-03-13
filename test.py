# 使用 timer_decorator 装饰一个简单的函数
import time

from utils.time_record import timer_decorator_with_info, timer_decorator


@timer_decorator
def example_function_1():
    time.sleep(2)  # 模拟一个耗时操作
    print("Function 1 executed")

# 使用 timer_decorator_with_info 装饰另一个函数
@timer_decorator_with_info("This is function 2 with additional info")
def example_function_2():
    time.sleep(1)  # 模拟一个耗时操作
    print("Function 2 executed")

# 调用被装饰的函数
example_function_1()
example_function_2()