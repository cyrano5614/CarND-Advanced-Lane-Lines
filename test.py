import threading

k = 10
x = 0
m = threading.Lock()

def foo():
	global x
	for i in xrange(k):
		with m:
			x += 1


def bar():
	global x
	for i in xrange(k):
		with m:
			x -= 1

t1 = threading.Thread(target=foo)
t2 = threading.Thread(target=bar)
t1.start()
t2.start()
t1.join()
t2.join()
print (x)