from collections import UserList


class FunctionStack(UserList):
    def free(self):
        #print("Freeing...")
        #print("Before:", self)
        #for obj in reversed(self):
        for obj in self:
            if obj.tainted:
                obj._touch()
            self.remove(obj)
        #if len(self) > 0:
        #    self.clear()
        #print("After:", self)
        #print("Freeing is finished!")

#_fstack = FunctionStack()
