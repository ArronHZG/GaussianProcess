import numpy as np

import matplotlib.pyplot as plt
from scipy.special import gamma
from sklearn import gaussian_process

# def plot_unit_gaussian_samples(D):
#     plt.title('sdf')
#
#     xs = np.linspace(0, 1, D)
#     for color in range(10):
#        ys = np.random.multivariate_normal(np.zeros(D), np.eye(D))
#        plt.plot(xs, ys)
#     return plt
# plot_unit_gaussian_samples(100).show()


def kernel(name,x,y):
    if name=='Linear kernel':
        return x*y
    elif name =='Brounian kernel':
        return min(x,y)
    elif name=='Squared exponential kernel':
        return np.exp(-100*(x-y)**2)
    elif name=='Ornstein–Uhlenbeck kernel':
        return np.exp(-100*np.fabs(x-y))
    # elif name=='Matérn kernel':
    #     return np.exp(1/gamma(1)*)
    elif name=='Rational quadratic kernel':
        return np.power(1+(x-y)**2,-10)

    elif name=='Periodic kernel':
        return np.exp(-np.sin(5*np.pi*(x-y))**2)

    elif name=='Symmetric kernel':
        return np.exp(-100*min(np.fabs(x-y),np.fabs(x+y))**2)


def plot_gaussian_samples(name,num,begin=-1,end=1,n=1000):
    x=np.linspace(begin,end,n).reshape((n,1))
    #covariance matrix
    cov=np.zeros((n,n))
    #kernal function
    for i in range(n):
        for j in range(n):
            cov[i][j]=kernel(name,x[i],x[j])
    plt.cla()
    plt.title(name)
    plt.ylim(-5, 5)#设置x轴的取值范围
    for _ in range(num):
        u = np.random.randn(n, 1)
        U, S, V = np.linalg.svd(cov)
        # print(np.dot(np.dot(U,np.diag(S)),V))
        z = np.dot(np.dot(U, np.sqrt(np.diag(S))), u)

        assert (x.shape == (n, 1))
        assert (z.shape == (n, 1))
        plt.plot(x, z)
    import os
    plt.savefig(os.path.join("pic",f"{name}-{num}.png"))

def plot_Squared_gaussian_sf(begin=-1,end=1,n=1000):
    ells=[0.01,0.1,1,5]
    l=0.5
    pic_num=0
    for ell in ells:
        pic_num += 1
        plt.subplot(2,2,pic_num)
        plt.ylim(-5,5)
        plt.subplots_adjust(wspace=0.5, hspace=0.5)  # 调整子图间距
        plt.title(f"σ={ell}")
        x = np.linspace(begin, end, n).reshape((n, 1))
        # covariance matrix
        cov = np.zeros((n, n))
        # kernal function
        for i in range(n):
            for j in range(n):
                cov[i][j] = ell * np.exp(-(x[i] - x[j]) ** 2 / (2 * l ** 2))
        for _ in range(5):
            u = np.random.randn(n, 1)
            U, S, V = np.linalg.svd(cov)
            # print(np.dot(np.dot(U,np.diag(S)),V))
            z = np.dot(np.dot(U, np.sqrt(np.diag(S))), u)
            assert (x.shape == (n, 1))
            assert (z.shape == (n, 1))
            plt.plot(x, z)
    import os
    plt.savefig(os.path.join("pic",f"Squared-sf.png"))



def plot_Squared_gaussian_l(begin=-1,end=1,n=1000):
    ell=5
    ls=[0.04,0.08,0.16,0.32]

    pic_num=0
    for l in ls:
        pic_num += 1
        plt.subplot(2,2,pic_num)
        plt.ylim(-9,9)
        plt.subplots_adjust(wspace=0.5, hspace=0.5)  # 调整子图间距
        plt.title(f"l={l}")
        x = np.linspace(begin, end, n).reshape((n, 1))
        # covariance matrix
        cov = np.zeros((n, n))
        # kernal function
        for i in range(n):
            for j in range(n):
                cov[i][j] = ell * np.exp(-(x[i] - x[j]) ** 2 / (2 * l ** 2))
        for _ in range(5):
            u = np.random.randn(n, 1)
            U, S, V = np.linalg.svd(cov)
            # print(np.dot(np.dot(U,np.diag(S)),V))
            z = np.dot(np.dot(U, np.sqrt(np.diag(S))), u)
            assert (x.shape == (n, 1))
            assert (z.shape == (n, 1))
            plt.plot(x, z)
    import os
    plt.savefig(os.path.join("pic",f"Squared-l.png"))


if __name__ == '__main__':
    # a=np.array([[1,2,3],[1,2,3],[1,2,3]])
    # u,s,v = np.linalg.svd(a)
    # print(np.dot(np.dot(u,np.diag(s)),v))
    # if name=='Linear kernel':
    #         return x*y
    #     elif name =='Brounian kernel':
    #         return min(x,y)
    #     elif name=='Squared exponential kernel':
    #         return np.exp(-(x-y)**2)
    #     elif name=='Ornstein–Uhlenbeck kernel':
    #         return np.exp(-np.fabs(x-y))
    #     # elif name=='Matérn kernel':
    #     #     return np.exp(-100*(x-y)**2)
    #     elif name=='Periodic kernel':
    #         return np.exp(-2*np.sin((x-y)/2)**2)
    #     elif name=='Rational quadratic kernel':
    #         return np.power(1+(x-y)**2,-1)
    num=10
    # plot_gaussian_samples('Linear kernel',num)
    # plot_gaussian_samples('Brounian kernel', num)
    # plot_gaussian_samples('Squared exponential kernel', 500)
    # plot_gaussian_samples('Ornstein–Uhlenbeck kernel', num)
    # # plot_gaussian_samples('Matérn kernel', 10)
    # plot_gaussian_samples('Periodic kernel', num)
    # plot_gaussian_samples('Rational quadratic kernel', num)
    # plot_gaussian_samples('Symmetric kernel', num)


    # plt.show()
    # plot_Squared_gaussian_sf()
    # plt.cla()
    # plot_Squared_gaussian_l()



