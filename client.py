from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import wl, wlexpr   # noqa

kernel_dir = (r"""/usr/local/Wolfram/WolframEngine/14.3/SystemFiles/Kernel/
              Binaries/Linux-x86-64/WolframKernel""")
session = WolframLanguageSession(kernel_dir)
session.start()
result = session.evaluate(wl.StringReverse('abc'))
print(result)
