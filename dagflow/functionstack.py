from collections import UserList


class FunctionStack(UserList):
    def free(self):
        print(self)
        for obj in reversed(self):
            obj(recursive=False)
            self.remove(obj)
        print(self)

_fstack = FunctionStack()
