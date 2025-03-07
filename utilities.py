# Std modules
from dataclasses import dataclass
import textwrap
import traceback
import os
os.system("") #For escape characters to work


def print_exception(e, function):
    msg = '\n'.join(textwrap.wrap(str(e), 80))
    bars = ''.join(['-']*80)
    s = ("{}{}{}\n".format(ESC.FAIL,bars,ESC.ENDC),
         "{}{}{}\n".format(ESC.FAIL,type(e).__name__,ESC.ENDC),
         "In function: ",
         "{}{}(){}\n".format(ESC.OKBLUE, function.__name__, ESC.ENDC),
         "{}{}{}\n".format(ESC.WARNING,  msg, ESC.ENDC),
         "{}{}{}".format(ESC.FAIL,bars,ESC.ENDC),)
    
    print("\n{}{}{}".format(ESC.FAIL, bars, ESC.ENDC))
    traceback.print_exc()
    print(''.join(s))

def safe_run(function, stop_function=None, args=(), stop_args=()):
    try: 
        function(*args)
        return 0
    except BaseException as e:
        if not stop_function is None: stop_function(*stop_args)
        print_exception(e, function)
        return -1


@dataclass
class ESC:
    HEADER: str = '\033[95m'
    OKBLUE: str = '\033[94m'
    OKCYAN: str = '\033[96m'
    OKGREEN: str = '\033[92m'
    WARNING: str = '\033[93m'
    FAIL: str = '\033[91m'
    ENDC: str = '\033[0m'
    BOLD: str = '\033[1m'
    UNDERLINE: str = '\033[4m'
    