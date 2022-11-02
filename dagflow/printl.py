# -*- coding: utf-8 -*-


printlevel = 0
singlemargin = "    "
marginflag = False
prefix_function = lambda: ""


def set_prefix_function(f):
    global prefix_function
    prefix_function = f


class next_level:
    def __enter__(self):
        global printlevel
        printlevel += 1

    def __exit__(self, *args, **kwargs):
        global printlevel
        printlevel -= 1


def current_level():
    return printlevel


def print_margin(kwargs):
    global marginflag
    prefix = kwargs.pop("prefix", prefix_function())
    postfix = kwargs.pop("postfix", None)
    prefixopts = kwargs.pop("prefixopts", dict(end=""))
    postfixopts = kwargs.pop("postfixopts", dict(end=" "))
    if marginflag:
        return

    if prefix:
        print(*prefix, **prefixopts)

    print(singlemargin * printlevel, sep="", end="")

    if postfix:
        print(*postfix, **postfixopts)

    marginflag = True


def reset_margin_flag(*args, **kwargs):
    global marginflag

    for arg in args + (kwargs.pop("sep", ""), kwargs.pop("end", "\n")):
        if "\n" in str(arg):
            marginflag = False
            return


def printl(*args, **kwargs):
    print_margin(kwargs)
    print(*args, **kwargs)
    reset_margin_flag(*args, **kwargs)
