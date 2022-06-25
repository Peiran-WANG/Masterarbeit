import streamlit as st
import numpy as np
import pandas as pd
import warnings
from PIL import Image 
from fractions import Fraction


st.markdown('结构健康监测决策系统')

# 设置网页标题
st.title('结构健康监测决策系统')

# 展示一级标题
st.header('1. 监测参数优选')

st.write('本步骤采用AHP分析方法,通过三个指标，进行权重对比，选择出合适的监测参数。若已获得监测参数，可直接进行第二步监测方法的决策')


image1 = Image.open('DSS1.png')
st.image(image1,caption='监测参数优选')


st.subheader('1.1 监测参数选择系统说明')

st.write('本步骤决策是基于层次分析法AHP, 这种方法是在同一层次影响因素重要性进行两两比较。衡量尺度划分为9个等级,如下图所示')
image2 = Image.open('AHP scale.png')
st.image(image2,caption='1-9 scale for AHP')

st.write('__*Parameter Sensitivity*__: ')
st.write('__*Damage correlation*__: ')
st.write('__*Monitoring economics*__')

st.subheader('1.2 重要性评估')

#  st.write('__评估“最佳参数选择”的相对重要性__')

transdict = {"1/9":"B绝对重要", "1/7":"B十分重要","1/5":"B比较重要","1/3":"B稍微重要", "1":"A和B同样重要","3":"A稍微重要", "5":"A比较重要","7":"A十分重要","9":"A绝对重要"}


# 第一个
st.write('1. 根据实际情况，您觉得**参数敏感性**和**损伤相关性**，对于最佳监测参数的选择哪个更重要')




DSS1_1 = st.select_slider("A:参数敏感性  B:损伤敏感性", 
                        ("1/9","1/7","1/5","1/3","1","3","5","7","9"),"1")
if DSS1_1:
    f"你觉得，参数敏感性与损伤敏感性相比，重要性为：{transdict[DSS1_1]}"

st.write('-----')

st.write('2. 根据实际情况，您觉得**参数敏感性**和**监测经济性**，对于最佳监测参数的选择哪个更重要')

DSS1_2 = st.select_slider("A:参数敏感性   B:监测经济性", 
                        ("1/9","1/7","1/5","1/3","1","3","5","7","9"),"1")
if DSS1_2:
    f"你觉得，参数敏感性与损伤敏感性相比，重要性为：{transdict[DSS1_2]}"

st.write('-----')

# 第二个
st.write('3. 根据实际情况，您觉得**损伤相关性**和**监测经济性**，对于最佳监测参数的选择哪个更重要')


DSS1_3 = st.select_slider("A: 损伤相关性  B:监测经济性", 
                        ("1/9","1/7","1/5","1/3","1","3","5","7","9"),"1")
if DSS1_3:
    f"**损伤相关性**和**监测经济性**相比，您觉得：{transdict[DSS1_3]}"

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
        assert n == n1, '不是一个方阵'  # 输入错误触发
        for i in range(n):
            for j in range(n):
                if np.abs(input_matrix[i, j] * input_matrix[j, i] - 1) > 1e-7:
                    raise ValueError('不是反互对称矩阵') # 不符合要求

        eigenvalues, eigenvectors = np.linalg.eig(input_matrix)  # 计算方阵的特征值和右特征向量。

        max_idx = np.argmax(eigenvalues) 
        # 函数功能，返回最大值的索引；若axis=1，表明按行比较，输出每行中最大值的索引，若axis=0，则输出每列中最大值的索引。
        max_eigen = eigenvalues[max_idx].real # 最大特征值
        eigen = eigenvectors[:, max_idx].real 
        eigen = eigen / eigen.sum()

        if n > 9:
            CR = None  
            warnings.warn('无法判断一致性')
        else:
            CI = (max_eigen - n) / (n - 1) 
            CR = CI / self.RI[n]   # Relative consistency index
        return max_eigen, CR, eigen  # 输出 最大特征值 CR  特征值

    def run(self):
        max_eigen, CR, criteria_eigen = self.cal_weights(self.criteria)
        st.write('**准则层**')
        st.write('最大特征值{:<5f}'.format(max_eigen))
        st.write('Relative consistency index (CR) = {:<5f}, 一致性检验{}通过'.format(CR, '' if CR < 0.1 else '不'))

        st.write('准则层权重 = {}\n'.format(criteria_eigen))  #格式化输出

        max_eigen_list, CR_list, eigen_list = [], [], []
        for i in self.b:
            max_eigen, CR, eigen = self.cal_weights(i)
            max_eigen_list.append(max_eigen)   # 在列表末尾添加新的对象。
            CR_list.append(CR)
            eigen_list.append(eigen)

        pd_print = pd.DataFrame(eigen_list,
                                index=['准则1','准则2','准则3' ],
                                columns=['方案1','方案2','方案3'],
                                )
        pd_print.loc[:, '最大特征值'] = max_eigen_list
        pd_print.loc[:, 'CR'] = CR_list
        pd_print.loc[:, '一致性检验'] = pd_print.loc[:, 'CR'] < 0.1
        
        
        st.write('**参数层**')
        st.table(pd_print)

        # 目标层
        obj = np.dot(criteria_eigen.reshape(1, -1), np.array(eigen_list))  # reshape 重塑
        obj1 = obj.flatten()


        st.write('\n**目标层**')
        st.write('参数权重 {}' .format(obj1))
        st.write('最优选择是参数 {}'.format(np.argmax(obj)+1))
        st.write('优选参数排序为 {}'.format(np.argsort(-obj1)+1))

        return obj





if st.button("进行决策评估"):
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
    b1 = np.array([[1, 1 / 3, 1 / 8], [3, 1, 1 / 3], [8, 3, 1]])
    b2 = np.array([[1, 2, 5], [1 / 2, 1, 2], [1 / 5, 1 / 2, 1]])
    b3 = np.array([[1, 1, 3], [1, 1, 3], [1 / 3, 1 / 3, 1]])
 
    b = [b1, b2, b3] # 方案
    st.write('-----')
    a = AHP(criteria, b).run()
    st.write('-----')


st.header('2. 监测方法选择')


