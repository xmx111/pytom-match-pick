# 导入os模块，用于与操作系统进行交互，例如文件路径操作
import os
# 导入sys模块，用于访问与Python解释器紧密相关的变量和函数
import sys


class mute_stdout_stderr:
    """
    上下文管理器，用于将标准输出（stdout）和标准错误输出（stderr）重定向到/dev/null。
    仅用于防止在单元测试中终端输出过多信息。
    """

    def __enter__(self):
        """
        进入上下文管理器时执行的操作。
        此方法会将标准输出和标准错误输出重定向到/dev/null。

        返回
        -------
        self: mute_stdout_stderr
            返回上下文管理器实例本身
        """
        # 打开/dev/null文件，以写入模式打开
        self.outnull = open(os.devnull, "w")
        # 保存当前的标准输出对象
        self.old_stdout = sys.stdout
        # 保存当前的标准错误输出对象
        self.old_stderr = sys.stderr
        # 将标准输出重定向到/dev/null
        sys.stdout = self.outnull
        # 将标准错误输出重定向到/dev/null
        sys.stderr = self.outnull
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        退出上下文管理器时执行的操作。
        此方法会将标准输出和标准错误输出恢复到原来的状态，并关闭/dev/null文件。

        参数
        ----------
        exc_type: type
            异常类型，如果没有异常则为None
        exc_val: Exception
            异常实例，如果没有异常则为None
        exc_tb: traceback
            异常的回溯信息，如果没有异常则为None
        """
        # 恢复标准输出到原来的状态
        sys.stdout = self.old_stdout
        # 恢复标准错误输出到原来的状态
        sys.stderr = self.old_stderr
        # 关闭/dev/null文件
        self.outnull.close()
        # 返回False，不抑制异常
        return False
