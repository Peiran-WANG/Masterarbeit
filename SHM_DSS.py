from platform import python_branch
from unittest import result
from nbformat import write
import streamlit as st
import numpy as np
import pandas as pd
import warnings
from PIL import Image 
from fractions import Fraction
from DSS_text import *

st.set_page_config(page_title="Structural Health Monitoring Decision Support System")

DSS_la = 0
Language  = ["English","Deutsch","中文"]	
########## 左栏
### 语言选择
choice_la = st.sidebar.selectbox("Select Language",Language)

if choice_la == "English":
    DSS_la = 0
elif choice_la == "Deutsch":
    DSS_la = 1
elif choice_la == "中文":
    DSS_la = 2

# 
las = DSS_la

### 内容选择

page = st.sidebar.selectbox(sbar_sbox_0[las],p_names[las])

### 姓名

st.sidebar.write(author[las])


########## 第一章 简介

if page == p_names[las][0]:

    st.title(k1t[las])

    st.write(k1_0[las])

    imagek1 = Image.open('Decision support system.png')
    st.image(imagek1,caption='Decision Support System')


    st.write(k1_1[las])
    st.write(k1_2[las])
    st.write(k1_3[las])



### AHP 
class AHP:
    def __init__(self, criteria, b): # 初始化
        self.RI = (0, 0, 0.58, 0.9, 1.12, 1.24, 1.32, 1.41, 1.45, 1.49)  # random consistency index
        self.criteria = criteria #准则 
        self.b = b  #方案
        self.num_criteria = criteria.shape[0] # 矩阵行数, [1] 矩阵列数  （准则个数）
        self.num_project = b[0].shape[0]   # 方案个数

    def cal_weights(self, input_matrix):
        input_matrix = np.array(input_matrix)
        n, n1 = input_matrix.shape            # n, n1 等于 输出矩阵长度
        assert n == n1, 'Not a square'  # 输入错误触发
        for i in range(n):
            for j in range(n):
                if np.abs(input_matrix[i, j] * input_matrix[j, i] - 1) > 1e-7:
                    raise ValueError('Not an inverse mutually symmetric matrix') # 不符合要求

        eigenvalues, eigenvectors = np.linalg.eig(input_matrix)  # 计算方阵的特征值和右特征向量。

        max_idx = np.argmax(eigenvalues) 
        # 函数功能，返回最大值的索引；若axis=1，表明按行比较，输出每行中最大值的索引，若axis=0，则输出每列中最大值的索引。
        max_eigen = eigenvalues[max_idx].real # 最大特征值
        eigen = eigenvectors[:, max_idx].real 
        eigen = eigen / eigen.sum()

        if n > 9:
            CR = None  
            warnings.warn('Unable to judge consistency')
        else:
            CI = (max_eigen - n) / (n - 1) 
            CR = CI / self.RI[n]   # Relative consistency index
        return max_eigen, CR, eigen  # 输出 最大特征值 CR  特征值

    def run(self):
        max_eigen, CR, criteria_eigen = self.cal_weights(self.criteria)
        st.write('**Criterion Layer**')
        st.write('Criterion Layer Weight = {}\n'.format(criteria_eigen))  #格式化输出
        st.write('Maximum Eigenvalue = {:<5f}'.format(max_eigen))
        st.write('Relative consistency index (CR) = {:<5f}'.format(CR))

        if CR < 0.1:
            st.success('Consistency check passed')
        else:
            st.error('Consistency check failed, please change the weights')
            return
            


        max_eigen_list, CR_list, eigen_list = [], [], []
        for i in self.b:
            max_eigen, CR, eigen = self.cal_weights(i)
            max_eigen_list.append(max_eigen)   # 在列表末尾添加新的对象。
            CR_list.append(CR)
            eigen_list.append(eigen)

        pd_print = pd.DataFrame(eigen_list,
                                index=['Parameter Sensitivity','Damage Correlation','Monitoring Economy'],
                                columns=['Stree-Strain','Vibration','Acoustic Wave','Impedance'],
                                )
        pd_print.loc[:, 'Maximum Eigenvalue'] = max_eigen_list
        pd_print.loc[:, 'CR'] = CR_list
        pd_print.loc[:, 'Consistency check'] = pd_print.loc[:, 'CR'] < 0.1
        
        
        st.write('**Parameter layer**')
        st.table(pd_print)

        # 目标层
        obj = np.dot(criteria_eigen.reshape(1, -1), np.array(eigen_list))  # reshape 重塑
        obj_weights = obj.flatten()
        obj_weights_list = list(obj_weights)

        tran ={0:"Stree-Strain",1:"Vibration",2:"Acoustic Wave",3:"Impedance"}
        objlist = list(np.argsort(-obj_weights))

        objmax_list = obj_weights_list.index(max(obj_weights_list))
        objmax = tran[objmax_list]

        objlist_t =list()
        for  item in objlist:
            i = tran[item]
            i = str(i)
            objlist_t.append(i)
        objlist_t1 = ',\n'.join(objlist_t)

        
        st.write('\n**Goal layer**')
        st.write('Parameter weights: {}' .format(obj_weights))
        st.write('The parameters are sorted as {}'.format(objlist_t1))
        st.info('The optimal parameters are: {}'.format(objmax))
        
        return obj

##########   第二章 监测参数选择

if page == p_names[las][1]:
    
    st.title(k2t[las])

    st.write(k2_1[las])

    imagek21 = Image.open('DSS1.png')
    st.image(imagek21,caption='AHP hierarchical structure')

    st.header(k2_2[las])

    st.write(k2_3_1[las])
    st.write(k2_3_2[las])
    st.write(k2_3_3[las])

    st.header(k2_4[las])
    
    st.write(k2_5[las])
    imagek22 = Image.open('AHP scale.png')
    st.image(imagek22,caption='1-9 importance scale')

    st.write('-----')
    st.subheader(k2_6[las])

    transdict = {"1/9":"**B** Absolute importance", "1/7":"**B** Very strong importance","1/5":"**B** Strong importance","1/3":"**B** Moderate importance", "1":"**A** and **B** Equal importance","3":"**A** Moderate importance", "5":"**A** Strong importance","7":"**A** Very strong importance","9":"**A** Absolute importance"}

    # 第一个

    DSS1_1 = st.select_slider("1. (A) Parameter Sensitivit compared to (B) Damage Correlation", 
                            ("9","7","5","3","1","1/3","1/5","1/7","1/9"),"1")
    if DSS1_1:
        f"Your choice is: {transdict[DSS1_1]}"

    st.write('-----')

    # 第二个

    DSS1_2 = st.select_slider("2. (A) Parameter Sensitivity compared to (B) Monitoring Economy", 
                            ("9","7","5","3","1","1/3","1/5","1/7","1/9"),"1")
    if DSS1_2:
        f"Your choice is: {transdict[DSS1_2]}"

    st.write('-----')

    # 第三个

    DSS1_3 = st.select_slider("3. (A) Damage Correlation compared to (B) Monitoring Economy", 
                            ("9","7","5","3","1","1/3","1/5","1/7","1/9"),"1")
    if DSS1_3:
        f"Your choice is: {transdict[DSS1_3]}"

    st.write('-----')

    if st.button(k2_7[las]):
        x1 = Fraction(DSS1_1)
        y1 = Fraction(DSS1_2)
        z1 = Fraction(DSS1_3)
        x = float(x1)
        y = float(y1)
        z = float(z1)


        criteria = np.array([[1, x, y],
                            [1 / x,1,z],
                            [1 / y, 1 / z, 1]])

        # 对每个准则，方案优劣排序 
        b1 = np.array([[1, 1, 1/2, 1/2], [1,1, 1/2, 1/2], [2, 2, 1,1],[2,2,1,1]])
        b2 = np.array([[1, 1/2,1/3,1/2], [2, 1, 1, 1], [3,1,1,1],[2,1,1,1]])
        b3 = np.array([[1, 1, 4,3], [1, 1, 3,2], [1/4, 1/3, 1,1],[1/3,1/2,1,1]])
        
    
        b = [b1, b2, b3] 
        a = AHP(criteria, b).run()




##################  第三章

if page == p_names[las][2]:
    
    st.title(k3t[las])

    st.write(k3_1[las])
    
    imagek31 = Image.open('RBR.png')
    st.image(imagek31,caption='Rule based reasoning decision system')


    st.header(k3_2[las])
    st.write(k3_21[las])

    imagek32 = Image.open('SHM Methods.png')
    st.image(imagek32,caption='SHM Methods')

    st.header(k3_3[las])
    st.write(k3_31[las])

    zuhe = list()


    transdict2 = {"Magnetic metal": 0 ,"Non-magnetic metal": 1 ,"Composites":2, "Ceramics":3, "Stree-Strain":4,"Vibration":5,"Acoustic Wave":6,"Impedance":7,"Time Domain":8,"Frequency Domain":9,"Time-frequency domain":10,"Global":11,"Local":12,"Global and Local":13,"Active":14,"Passive":15,"Detection":16,"Location":17,"Assessment":18,"Prediction":19,"Identification Accuracy":20,"Environmental Factor":21,"Cost Effectiveness":22,"Either one":23}

    ##############
    st.subheader('1. Please select the material of hybrid structure')

    jinshu = st.selectbox("The metal materials in the hybrid structure is",
                        ("Magnetic metal","Non-magnetic metal"))


    feijinshu = st.selectbox("The non-metallic materials in the hybrid structure is",
                        ("Composites","Ceramics"))

    st.info("The hybrid structural materials  are {} and {} ".format(jinshu,feijinshu))

    zuhe.append(transdict2[jinshu])
    zuhe.append(transdict2[feijinshu])
    st.write('-----')

    ##################################
    st.subheader('2. Please select monitoring parameters')

    canshu = st.selectbox("The monitoring parameter of this monitoring system is:",
                        ("Stree-Strain","Vibration","Acoustic Wave","Impedance"))

    st.info("The monitored parameter is  {}".format(canshu))

    zuhe.append(transdict2[canshu])
    st.write('-----')

    ###################################

    st.subheader('3. Please select the data processing method')
    chuli = st.selectbox("The system's data processing method is:",
                        ("Time Domain","Frequency Domain","Time-frequency domain","Either one"))

    st.info("The system's data processing method is {}".format(chuli))

    zuhe.append(transdict2[chuli])

    st.write('-----')
    ###################################

    st.subheader('4. Please select the Monitoring range')

    fanwei = st.selectbox("The Monitoring range is:",
                        ("Global","Local","Global and Local","Either one"))


    st.info("The system's Monitoring range is: {}".format(fanwei)) 

    zuhe.append(transdict2[fanwei])
    st.write('-----')

    ###################################


    st.subheader('5. Please select the Monitoring mode')


    fangshi = st.selectbox("The system monitoring type is:",
                        ("Active","Passive","Either one"))

    st.info("The system's monitoring mode is: {}".format(fangshi)) 

    zuhe.append(transdict2[fangshi])
    st.write('-----')

    ###################################


    st.subheader('6. Please select the Damage analysis function')

    fenxi = st.multiselect("The system damage analysis functions are:",
                        options = ("Detection","Location","Assessment","Prediction"),
                        default = ("Detection"))
    fenxi2 = ',\n'.join(fenxi)
    st.info("The system's Damage analysis functions are:  {}".format(fenxi2))

    fenxi1 = list()
    for  item in fenxi:
        i = transdict2[item]
        i = int(i)
        fenxi1.append(i)

    zuhe.extend(fenxi1)
    st.write('-----')

    ###################################


    st.subheader('7. Please select other influence factors')

    qita = st.multiselect("Other factors of the system:",
                        options = ("Identification Accuracy","Environmental Factor","Cost Effectiveness"),
                        default = ("Identification Accuracy"))
    qita1 = ',\n'.join(qita)
    st.info("Other factors of the system are: {}".format(qita1))

    qita2 = list()
    for  item in qita:
        i = transdict2[item]
        i = int(i)
        qita2.append(i)

    zuhe.extend(qita2)

    ###################################

    a0 = np.array([0,0,0,0,  -10,1,-10,-10,  0,0,0,  1,-10,-10,  -10,1,   0,-10,0,0,   -10,-10,1,0])
    a1 = np.array([0,0,0,0,  -10,1,-10,-10,  0,0,0,  -10,1,-10,  0,1,     0,1,0,0,      1,0,0,0])
    a2 = np.array([0,0,0,0,  -10,1,-10,-10,  0,0,0,  0,0,1,     0,1,      0,1,0,0,     0,1,0,0])
    a3 = np.array([0,0,0,0,   1,-10,-10,-10,  0,0,0,  0,1,0,     0,0,      0,0,0,0,     1,0,0,0])
    a4 = np.array([0,0,0,0,  -10,-10,-10,1,  0,0,0,  1,0,0,     1,0,      0,0,0,0,     0,1,0,0])
    a5 = np.array([0,0,0,0,  -10,-10,1,-10,  0,0,0,  0,0,1,     1,0,      0,1,0,0,     1,1,1,0])
    a6 = np.array([0,0,0,0,     0,0,0,0,     0,0,0,  0,0,0,      0,0,      0,0,0,1,    0,0,-10,0])


    trans = {0:"Natural Frequency",1:"Mode Shape-Curvature",2:"Frequency response function",3:"Stress based",4:"Electro Mechanical Impedance",5:"Lamb Wave",6:"Artificial Intelligence"}

    st.write('-----')

    if st.button(k3_6[las]):

        a0sum = sum(a0.take(zuhe))
        a1sum = sum(a1.take(zuhe))
        a2sum = sum(a2.take(zuhe))
        a3sum = sum(a3.take(zuhe))
        a4sum = sum(a4.take(zuhe))
        a5sum = sum(a5.take(zuhe))
        a6sum = sum(a6.take(zuhe))

        st.write('-----')


        asum = [a0sum,a1sum,a2sum,a3sum,a4sum,a5sum,a6sum]

        ## 方法排序
        alist = np.array(asum)
        a_list_1 = list(np.argsort(-alist))
        a_list_t =list()
        for  item in a_list_1:
                i = trans[item]
                i = str(i)
                a_list_t.append(i)
        a_list_t1 = ',  '.join(a_list_t)

        ##### 排名
        st.subheader(k3_4[las])

        st.write('{}'.format(a_list_t1))

        st.write('-----')
        # max
        amax = asum.index(max(asum))
        amax_t = trans[amax]



        ## 坏方案
        bad = list()

        for item in asum:
            if item < -10:
                i = asum.index(item)
                asum[i]=0
                i_t = trans[i]
                bad.append(i_t)
                

        
        bad1 = ', '.join(bad)

        st.subheader("Decision result for monitoring method")
        st.error('Non-recommended monitoring methods: {}'.format(bad1))
        st.success('Optimal monitoring method is: {}'.format(amax_t))

        st.write('-----')

        st.subheader(k3_5[las])

        st.write(k3_51[amax])
        st.write(k3_52[amax])

        st.write('-----')
