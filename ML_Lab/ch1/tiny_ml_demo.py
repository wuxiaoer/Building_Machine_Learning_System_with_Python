#coding:utf-8
import scipy as sp
import matplotlib.pyplot as plt

#读取数据
data = sp.genfromtxt("./data/web_traffic.tsv", delimiter="\t")

#debug
print(data[:10])
print(data.shape)

x = data[:, 0]
y = data[:, 1]

#debug
print(sp.sum(sp.isnan(y)))

x = x[~sp.isnan(y)]
y = y[~sp.isnan(y)]

# plot the (x,y) points with dots of size 10
plt.scatter(x, y, s = 10)
plt.title("Web traffic over the last month")
plt.xlabel("Time")
plt.ylabel("Hits/hour")
plt.xticks([w*7*24 for w in range(10)], ['week %i' % w for w in range(10)])
plt.autoscale(tight = True)

# draw a slightly opaque, dashed grid
plt.grid(True, linestyle = '-', color = '0.75')
#plt.show()

def error(f, x, y):
    return sp.sum((f(x) - y) ** 2)

fp1, residuals, rank, sv, rcond = sp.polyfit(x, y, 1, full = True)
# debug
print("Model parameters: %s" % fp1)

f1 = sp.poly1d(fp1)
print(error(f1, x, y))

fx = sp.linspace(0, x[-1], 1000)
plt.plot(fx, f1(fx), "r-", label = "d=1", linewidth = 4)
#plt.legend(["d=%i" % f1.order], loc="upper left")
#plt.show()

f2p = sp.polyfit(x, y, 2)
print(f2p)
f2 = sp.poly1d(f2p)
print(error(f2, x, y))

plt.plot(fx, f2(fx),"b-", label = "d=2", linewidth = 4)
#plt.legend(["d=%i" % f2.order], loc="upper left")
#plt.show()

f3p = sp.polyfit(x, y, 3)
print(f3p)
f3 = sp.poly1d(f3p)
print(error(f3, x, y))

plt.plot(fx, f3(fx), "g--", label = "d=3", linewidth = 4)
#plt.legend(["d=%i" % f3.order], loc="upper left")
plt.legend(loc="upper left")
#plt.show()

f10p = sp.polyfit(x, y, 10)
print(f10p)
f10 = sp.poly1d(f10p)
print(error(f10, x, y))

plt.plot(fx, f10(fx), "r-", label = "d=10", linewidth = 4)
#plt.legend(["d=%i" % f10.order], loc="upper left")
plt.legend(loc="upper left")
#plt.show()


f53p = sp.polyfit(x, y, 53)
print(f53p)
f53 = sp.poly1d(f53p)
print(error(f53, x, y))

plt.plot(fx, f53(fx), "g-", label = "d=53", linewidth = 4)
#plt.legend(["d=%i" % f53.order], loc="upper left")
plt.legend(loc="upper left")
plt.show()
