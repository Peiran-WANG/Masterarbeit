from platform import python_branch
from unittest import result
from nbformat import write
import streamlit as st
import numpy as np
import pandas as pd
import warnings
from PIL import Image 
from fractions import Fraction


st.markdown('Masterarbeit- Development of a concept for data-based condition monitoring and structural health monitoring strategies for the use phase')

# 设置网页标题
st.title('Structural Health Monitoring System Decision Support System')

st.write('The decision system developed in this thesis is classified in two steps. Figure shows the framework of the decision support system.')
image1 = Image.open('Decision support system.png')
st.image(image1,caption='Monitoring Decision Support System')



# 展示一级标题
st.header('1. Selection of optimal monitoring parameters')

st.write('In this step, Analytic Hierarchy Process (AHP) is used to select the suitable monitoring parameters with three criterions of **Parameter Sensitivity**, **Damage Correlation** and **Monitoring Economy**, followed by a comparison of two weights.')
st.write('If the monitoring parameters have already been determined, the decision on the monitoring method can be made directly in the second step.')

image1 = Image.open('DSS1.png')
st.image(image1,caption='AHP hierarchical structure')


st.subheader('1.1 Monitoring parameters selection instructions')

st.write('This step of decision making is based on the Analytic Hierarchy Process (AHP), which is a two-by-two comparison of the importance of influencing factors at the same level. The measurement scale is divided into 9 levels, as shown below')
image2 = Image.open('AHP scale.png')
st.image(image2,caption='1-9 scale for AHP')

st.write('__*Parameter Sensitivity*__: Reflects the sensitivity of the monitoring parameters to the degradation process of the system health condition, the more sensitive the parameters are, the easier it is to capture the performance degradation state of the structure')
st.write('__*Damage correlation*__: Monitoring parameters can reflect multiple health state performance of the structure (reflecting the type of damage).')
st.write('__*Monitoring economics*__: The cost price to be paid for monitoring, including monitoring cost, information transmission and processing cost, operation cost and labor cost, etc.')

st.subheader('1.2 criterion Importance Assessment')

#  st.write('__评估“最佳参数选择”的相对重要性__')

transdict = {"1/9":"**B** Absolute importance", "1/7":"**B** Very strong importance","1/5":"**B** Strong importance","1/3":"**B** Moderate importance", "1":"**A** and **B** Equal importance","3":"**A** Moderate importance", "5":"**A** Strong importance","7":"**A** Very strong importance","9":"**A** Absolute importance"}


# 第一个
st.write('1. For monitoring systems, which do you think is more important,  **A: Parameter Sensitivity** or **B: Damage Correlation** of monitoring parameters?')

DSS1_1 = st.select_slider("A: Parameter Sensitivity vs  B: Damage Correlation", 
                        ("9","7","5","3","1","1/3","1/5","1/7","1/9"),"1")
if DSS1_1:
    f"**A: Parameter Sensitivity** compared to **B: Damage Correlation**, do you think: {transdict[DSS1_1]}"

st.write('-----')

st.write('2. For monitoring systems, which do you think is more important,   **A: Parameter Sensitivity** or **B: Monitoring Economy** of monitoring parameters?')

DSS1_2 = st.select_slider("A: Parameter Sensitivity vs Monitoring Economy", 
                        ("9","7","5","3","1","1/3","1/5","1/7","1/9"),"1")
if DSS1_2:
    f"**A: Parameter Sensitivity** compared to **B: Monitoring Economy**, do you think: {transdict[DSS1_2]}"

st.write('-----')

# 第二个
st.write('3. For monitoring systems, which do you think is more important,   **A: Damage Correlation** or **B: Monitoring Economy** of monitoring parameters?')


DSS1_3 = st.select_slider("A: Damage Correlation vs Monitoring Economy", 
                        ("9","7","5","3","1","1/3","1/5","1/7","1/9"),"1")
if DSS1_3:
    f"**A: Damage Correlation** compared to **B: Monitoring Economy**, do you think: {transdict[DSS1_3]}"

st.write('-----')

# AHP 






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
            

        st.write('Criterion Layer Weight = {:<5f}\n'.format(criteria_eigen))  #格式化输出

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





if st.button("Click for monitoring parameter selection"):
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
    b1 = np.array([[1, 3, 1/2, 1], [1/3,1, 1/4, 1/3], [2, 4, 1,1],[1,3,1,1]])
    b2 = np.array([[1, 1/3,1/3,1], [3, 1, 1, 3], [3,1,1,3],[1,1/3,1/3,1]])
    b3 = np.array([[1, 1/3, 3,2], [3, 1, 3,2], [1/3, 1/3, 1,1],[1/2,1/2,1,1]])
    
 
    b = [b1, b2, b3] 
    a = AHP(criteria, b).run()
    



code1 ='''
class AHP:
    def __init__(self, criteria, b): 
        self.RI = (0, 0, 0.58, 0.9, 1.12, 1.24, 1.32, 1.41, 1.45, 1.49)  # random consistency index
        self.criteria = criteria  
        self.b = b  
        self.num_criteria = criteria.shape[0] 
        self.num_project = b[0].shape[0]   

    def cal_weights(self, input_matrix):
        input_matrix = np.array(input_matrix)
        n, n1 = input_matrix.shape           
        assert n == n1, 'Not a square'  
        for i in range(n):
            for j in range(n):
                if np.abs(input_matrix[i, j] * input_matrix[j, i] - 1) > 1e-7:
                    raise ValueError('Not an inverse mutually symmetric matrix') 

        eigenvalues, eigenvectors = np.linalg.eig(input_matrix)  

        max_idx = np.argmax(eigenvalues) 
       
        max_eigen = eigenvalues[max_idx].real 
        eigen = eigenvectors[:, max_idx].real 
        eigen = eigen / eigen.sum()

        if n > 9:
            CR = None  
            warnings.warn('Unable to judge consistency')
        else:
            CI = (max_eigen - n) / (n - 1) 
            CR = CI / self.RI[n]   # Relative consistency index
        return max_eigen, CR, eigen  

    def run(self):
        max_eigen, CR, criteria_eigen = self.cal_weights(self.criteria)
        st.write('**Criterion Layer**')
        st.write('Maximum Eigenvalue = {:<5f}'.format(max_eigen))
        st.write('Relative consistency index (CR) = {:<5f}'.format(CR))
        if CR < 0.1:
            st.success('Consistency check passed')
        else:
            st.error('Consistency check failed, please change the weights')
            return
            

        st.write('Criterion Layer Weight = {}\n'.format(criteria_eigen))  

        max_eigen_list, CR_list, eigen_list = [], [], []
        for i in self.b:
            max_eigen, CR, eigen = self.cal_weights(i)
            max_eigen_list.append(max_eigen)   
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
        obj = np.dot(criteria_eigen.reshape(1, -1), np.array(eigen_list))  # reshape 
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





if st.button("Click for monitoring parameter selection"):
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
    b1 = np.array([[1, 3, 1/2, 1], [1/3,1, 1/4, 1/3], [2, 4, 1,1],[1,3,1,1]])
    b2 = np.array([[1, 1/3,1/3,1], [3, 1, 1, 3], [3,1,1,3],[1,1/3,1/3,1]])
    b3 = np.array([[1, 1/3, 3,2], [3, 1, 3,2], [1/3, 1/3, 1,1],[1/2,1/2,1,1]])
    
 
    b = [b1, b2, b3] 
    a = AHP(criteria, b).run()
'''
st.write('-----')
with st.expander("Click to view decision codes",False):
    st.code(code1, language="python")



###################################################################

st.header('2. Selection of the optimal monitoring methods')

st.write('This step uses a decision method based on literature knowledge base reasoning, which analyzes and summarizes various monitoring methods based on information related to the monitoring system through literature research, and produces a knowledge base to help decision makers make judgments based on the actual situation')

zuhe = list()


transdict2 = {"Magnetic metal": 0 ,"Non-magnetic metal": 1 ,"Polymers": 2,"Composites":3, "Ceramics":4, "Stree-Strain":5,"Vibration":6,"Acoustic Wave":7,"Impedance":8,"Time Domain":9,"Frequency Domain":10,"Time-frequency domain":11,"Global":12,"Local":13,"Active":14,"Passive":15,"Degree of damage":16,"Damage localization":17,"Life prediction":18,"Good accuracy":19,"Low environmental impact":20,"Good economic":21,"Either one":22}


#################################
st.subheader('2.1 Please select the material of hybrid structure')

jinshu = st.selectbox("The metal materials in the hybrid structure is",
                    ("Magnetic metal","Non-magnetic metal"))


feijinshu = st.selectbox("The non-metallic materials in the hybrid structure is",
                    ("Polymers","Composites","Ceramics"))

st.info("The hybrid structural materials  are {} and {} ".format(jinshu,feijinshu))

zuhe.append(transdict2[jinshu])
zuhe.append(transdict2[feijinshu])
st.write('-----')

##################################

st.subheader('2.2 Please select monitoring parameters')

canshu = st.selectbox("The monitoring parameter of this monitoring system is:",
                    ("Stree-Strain","Vibration","Acoustic Wave","Impedance"))

st.info("The monitored parameter is  {}".format(canshu))

zuhe.append(transdict2[canshu])
st.write('-----')

###################################

st.subheader('2.3 Please select the data processing method')
chuli = st.selectbox("The system's data processing method is:",
                    ("Time Domain","Frequency Domain","Time-frequency domain","Either one"))

st.info("The system's data processing method is {}".format(chuli))

zuhe.append(transdict2[chuli])

st.write('-----')
###################################


st.subheader('2.4 Please select the system monitoring range')

fanwei = st.selectbox("The system monitors the range is:",
                    ("Global","Local","Either one"))


st.info("The system monitors the range is: {}".format(fanwei)) 

zuhe.append(transdict2[fanwei])
st.write('-----')

###################################


st.subheader('2.5 Please select the monitoring type')


fangshi = st.selectbox("The system monitoring type is:",
                    ("Active","Passive","Either one"))

st.info("The system monitoring type is: {}".format(fangshi)) 

zuhe.append(transdict2[fangshi])
st.write('-----')

###################################


st.subheader('2.6 Please select the damage analysis function')

fenxi = st.multiselect("The system damage analysis functions are:",
                    options = ("Degree of damage","Damage localization","Life prediction"),
                    default = ("Degree of damage"))
fenxi2 = ',\n'.join(fenxi)
st.info("The system damage analysis functions are:  {}".format(fenxi2))

fenxi1 = list()
for  item in fenxi:
    i = transdict2[item]
    i = int(i)
    fenxi1.append(i)

zuhe.extend(fenxi1)
st.write('-----')

###################################


st.subheader('2.7 Please select other influence factors')

qita = st.multiselect("Other factors of the system:",
                    options = ("Good accuracy","Low environmental impact","Good economic"),
                    default = ("Good accuracy"))
qita1 = ',\n'.join(qita)
st.info("Other factors of the system are: {}".format(qita1))

qita2 = list()
for  item in qita:
    i = transdict2[item]
    i = int(i)
    qita2.append(i)

zuhe.extend(qita2)

###################################



a0 = np.array([0,0,0,0,0,  -10,1,-10,-10,  0,0,0,  1,-10,  -10,1,  0,-10,0,   -10,-10,1,0])
a1 = np.array([0,0,0,0,0,  -10,1,-10,-10,  0,0,0,  -10,1,  0,0,    0,0,0,      1,0,0,0])
a2 = np.array([0,0,0,0,0,  1,-10,-10,-10,  0,0,0,  0,1,    0,0,    0,0,0,     0,1,0,0])
a3 = np.array([0,0,0,0,0,  -10,1,-10,-10,  0,0,0,  1,0,    0,0,    0,0,0,     0,0,0,0])
a4 = np.array([0,0,0,0,0,  -10,-10,-10,1,  0,0,0,  0,0,    1,0,    0,0,0,     0,1,0,0])
a5 = np.array([0,0,0,0,0,  -10,-10,1,-10,  0,0,0,  0,0,    1,0,    0,1,0,     1,1,1,0])
a6 = np.array([0,0,0,0,0,     0,0,0,0,      0,0,0,  0,0,    0,0,   0,0,1,     0,0,0,0])



trans = {0:"Natural Frequency",1:"Mode Shape-Curvature",2:"Modal strain energy",3:"Frequency response function",4:"Electro Mechanical Impedance",5:"Lamb Wave",6:"Artificial Intelligence"}

st.write('-----')

if st.button("Click for monitoring method selection"):

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

    st.subheader('The monitoring methods are ordered by requirement as: ')
    st.write('{}'.format(a_list_t1))


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

    st.error('Non-recommended monitoring methods: {}'.format(bad1))
    st.success('Optimal monitoring method is: {}'.format(amax_t))

    st.write('-----')
