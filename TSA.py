import numpy as np
import pandas as pd
from scipy import stats
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")
import math
from decimal import *
getcontext().prec = 1

class ts_sample:
    def __init__(self,data):
        self.name='Time Series from sampling data'
        self.data =data
        self.acf = self.acf()[1:]
        self.pacf = self.pacf_kk(self.acf)

    def plot(self):
        data = self.data
        plt.figure(figsize = (14,6))
        plt.plot(data, color = 'black',marker=".")
        plt.show()

    def acf(self):
        """
        sample autocorrelation function
        """
        tsdata = self.data
        n = len(tsdata)
        k = n #if n <20 else 20
        acf_list = []

        varz = sum((tsdata-np.mean(tsdata))**2)
        barz = np.mean(tsdata)
        for i in range(k):
            zt = tsdata[0:n-i]
            zt_k = tsdata[i:]
            rk = sum((zt-barz)*(zt_k-barz))/varz
            acf_list.append(rk)
        #print(acf_list)
        return acf_list

    def pacfk(self,acf_list,k):
        """
        use acf to culculate partial autocorrelation Yule-Walker equations
        """
        n = len(acf_list)
        rho_v = np.array(acf_list)
        rho_m = np.ones((n,n))
        for i in range(n):
            for j in range(n):
                if i ==j : rho_m[i,j]=1
                elif i < j: rho_m[i,j] = acf_list[j-i-1]
                elif i >j : rho_m[i,j] = acf_list[i-j-1]
        #print(rho_m,rho_v)
        #print(np.linalg.inv(rho_m[0:k,0:k]))
        pacf_v = np.dot(np.linalg.inv(rho_m[0:k,0:k]),rho_v[:k].reshape(-1,1))
        #print(pacf_v)
        return pacf_v

    def pacf_kk(self,acf_list):
        pacf_kk_list = []
        for i in range(1,len(acf_list)+1):
            pacf_list = self.pacfk(acf_list,i)
            pacf_kk_list.append(pacf_list[-1])
        return pacf_kk_list

class Analysis:
    def _init_(self):
        self.name = ''

    def Ljung_Box(residual_acf,nparameter,nsize,k,alpha=0.05):
        """
        input:list of residual acf, number of the parameters, size of the data, K lag and the credible level alpha
        Output:a dictionary:{'Q_values','P_value','test_result'}
        """
        dt = np.dtype(np.float32)
        residual_acf = np.array(residual_acf,dtype=dt)
        Q = nsize*(nsize+2)
        x=0
        for i in range(k):
            x +=(residual_acf[i]**2)/(nsize-1-i)
        Q = np.round(Q*x,4)
        P = np.round(stats.chi2.pdf(Q, k-nparameter),4)
        test = "accept" if P<alpha else "reject"
        line = "-"*40
        print(line)
        print("| Ljung_Box_test",'n=%d'%nsize,"K=%d"%k,"m=%d"%nparameter)
        print(line)
        print("| Q_value:","|",Q)
        print(line)
        print("| P_value ","|",1-P)
        print(line)
        print("| test    ","|",test)
        print(line)
        print("null hypothesis: the error terms have no correlation")
        return {'Q_values':Q,'P_value':1-P,'test_result':test}

    def Bartlets_apprx(acf_list,nsize,acftype='acf',k=20,alpha=0.05):
        """
        input: acf list, size of data, k lags, confidence level
        output: Bartlet's approximation
        """
        B_list = []
        z = stats.norm.ppf(alpha/2)
        for m in range(k):
            x = 1
            for i in range(m):
                x += 2*(acf_list[i]**2)

            B = np.round(z/(nsize**(0.5))*(x**(0.5)),4)
            B_interval = (-B,B) if B>0 else (B,-B)
            B_list.append(B_interval)
        fig = plt.figure(figsize=(12,5))
        plt.plot((0,len(acf_list)+1),(0,0),color = 'black')
        for i in range(len(acf_list)):
            plt.plot((i+1,i+1),(0,acf_list[i]),color="blue")
            plt.plot(i+1,acf_list[i],color="blue",marker="o")
        #print(" Bartlet's approximation:", B_list)
        plt.plot([int(i+1) for i in range(len(B_list))],[i[1] for i in B_list],color = 'red')
        plt.plot([int(i+1) for i in range(len(B_list))],[i[0] for i in B_list],color = 'red')
        plt.title(acftype.upper()+"& Bartlet's approximation")
        plt.ylabel(acftype.upper());plt.xlabel('Lag')
        return B_list


class ARMAseries:
    def __init__(self,phi=[],theta=[]):
        """
        we build a object here which we input theta and phi into it;
        and we detect what series it is
        """
        self.phi = phi
        self.theta = theta
        self.p = len(phi); self.q = len(theta)
        self.type = "AR(%d)MA(%d)"%(self.p,self.q)

    def acf(self):
        """
        In this function we try to print the acf and avcf out;
        the k denotes tht log and it defult to be 200 of which the
        acf would mostly be 0 normally
        But Now Only acf of MA model was Made..........Wait for updating
        """
        theta = self.theta
        phi = self.phi
        # make a list of acf
        acf = []
        ############ if it is a MA(q)modle:######################
        if self.p == 0 and self.q>0:
            # function gamma_k(theta):
            def cul_gamma(theta,q,k=100):
                matrix = np.zeros((q+1,q+1));matrix[0,0] = 1
                for i in range(q):matrix[0,i+1] = -theta[i]
                for i in range(q):
                    for k in range(q-i):
                        matrix[i+1,k] = -theta[i]*-theta[i+k]
                """
                we build a matrix like this:
                |1        |-theta1        |-theta2|
                |theta1**2|-theta1*-theta2|   0   |
                |theta2**2|      0        |   0   |
                gamma_0 = sum(matrix[:,0])
                gamma_1 = sum(matrix[:,1])
                gamma_2 = sum(matrix[:,1])
                """
                gamma=[]
                for k in range(q+1):
                    gamma_k = 0
                    for i in range(q+1):
                        gamma_k += matrix[i,k]
                    gamma.append(gamma_k)
                return gamma


            gamma = cul_gamma(theta,self.q)
            for i in gamma: acf.append(i/gamma[0])

        ############## if it is a AR(p) model#######################
        # if self.p > 0 and self.q==0:
        ############## if it is a ARMA(p,q) model #################
        return {'acf':acf,"acvf":gamma}

    def prediction(self,steps,Z,a,uptopoint,sigma2=1,alpha=0.05,intercept=None):
        """
        Input:
            steps: How many steps you want to predict;
            Z: list of  Zt in order like [Z1,Z2,Z3];
            a: list of error term in order [a1,a2,a3];
            uptopoint: if Z3 were the lastest record uptopoint = 3;
            sigma2: variance of error term;
            alpha: credible level for predictions
        Output:
            pandas.DataFrame of predictions with colums of 'prediction','var','ci_low','ci_upper'

        """
        if intercept != None:
            Z = [Z[i]-intercept for i in range(len(Z))]

        an = stats.norm(0,sigma2**(0.5))
        Zl = np.zeros((1,uptopoint)).tolist()[0]
        al = np.zeros((1,uptopoint)).tolist()[0]
        Zl[:len(Z)] = Z[::-1]
        al[:len(a)] = a[::-1]

        al_mean = al.copy() # for E(a) calculations
        al_var = np.zeros((1,uptopoint)).tolist()[0] # for Var(a) calculation

        predictions = pd.DataFrame(columns=['prediction','var','ci_low','ci_upper'])

        phi = np.array(self.phi); theta = np.array(self.theta)

        theta2 = np.append(1,theta*-1)
        theta2 = np.append(theta2,np.zeros((1,steps)))  # [1, -theta1, -theta2, ... , -thetaq]
        phi2 = np.append(1,phi) # [1,phi1,phi2,..,phip]

        mphi = np.zeros((len(theta2),len(phi2)+len(theta2)-1)).tolist()
        mphi[0][:len(phi2)] = phi2
        a_parameter = np.zeros((len(phi),len(theta2)))
        for i in range(steps):
            #############################################
            # calculate the mean of prediction:
            ar = phi*Zl[:len(phi)]              # AR part
            ma = -theta*al_mean[:len(theta)]    # MA part
            an = an
            EZl = np.round(np.sum(ar)+ np.sum(ma)+an.mean(),5)

            Zl.insert(0,EZl)
            al.insert(0,an)
            al_var.insert(0,an.var())
            al_mean.insert(0,an.mean())

            ##############################################
            # calculate the var of predictions:
            #The method is shown below:
            #                              iteration1:                                      iteration2:
            #           a_n(l)   a_n(l-1)  a_n(l-2)  ... a_n  a_n-1         a_n(l+1)   a_n(l)  a_n(l-1)  ... a_n  a_n-1
            #            1       -theta1   -theta2        0     0             1       -theta1   -theta2        0     0
            #                                 +                                                      +
            # phi1 *     0          0         0           0     0             0          1          x1        x2    xj
            #                                 +                                                      +
            # phi2 *     0          0         0           0     0             0           0          0         0     0
            #                                 =                                                      =
            #            1          x1        x2          xj    xj-1          1          x*1        x*2       x*j   x*j-1

            v = [phi[i]*a_parameter[i] for i in range(len(phi))]
            v = np.sum(v,axis=0)
            v = v+theta2
            variance = np.round(sum(v**2*al_var[:len(v)]),5)
            for j in range(len(v)):
                if al_var[j] == 0:
                    v[j]=0    # rid the parameter of deterministic errro terms
            v = v.tolist()
            v.pop()
            v.insert(0,0) # for next error predicted
            a_parameter = a_parameter.tolist()
            a_parameter.pop()
            new = a_parameter[0][:-1];new.insert(0,0)# for next error predicted
            a_parameter[0] = new # for next error predicted
            a_parameter.insert(0,v) # a_parameter treat it as a stack frist in later out
            a_parameter = np.array(a_parameter)
            # calculate the interval
            if intercept != None:
                EZl = EZl+intercept
            rv = stats.norm(0,1)
            Z_alpha = np.round(rv.ppf(1-alpha/2),2)
            ci = (EZl-Z_alpha*np.sqrt(variance),EZl+Z_alpha*np.sqrt(variance))
            print('Zl:',i+1,'| mean:', EZl," | Variance:" ,variance , " | Credible interval:", ci)
            prediction = pd.DataFrame([[EZl,variance,ci[0],ci[1]]], columns=['prediction','var','ci_low','ci_upper'],index=[uptopoint+1+i])
            predictions=predictions.append(prediction)
        return predictions

    def generate_serise(self,n,sigma2):
        """

        """
        a_stack = np.zeros((1,n))[0]

        Z_stack = np.zeros((1,n))[0]

        theta = np.array(self.theta);phi = np.array(self.phi)
        rv = stats.norm(0,sigma2**0.5)
        for i in range(n):
            ar = phi * Z_stack[:len(phi)]
            ma = -theta * a_stack[:len(theta)]
            ai = rv.rvs()
            zi = np.sum(ar)+np.sum(ma) + ai
            a_stack = a_stack.tolist()
            a_stack.pop()
            a_stack.insert(0,ai)
            a_stack = np.array(a_stack)
            Z_stack = Z_stack.tolist()
            Z_stack.pop()
            Z_stack.insert(0,zi)
            Z_stack = np.array(Z_stack)
        Z_stack = Z_stack.tolist() ; a_stack = a_stack.tolist()
        return Z_stack[::-1], a_stack[::-1]
