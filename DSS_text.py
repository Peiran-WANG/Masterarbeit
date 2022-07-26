
# 0 for english , 1 for deutsch， 2 for chinese

sbar_sbox_0 = ("Please Select A Chapter","Bitte wählen Sie ein Kapitel","请选择一个章节")
author = ("Autor: Peiran WANG","Autor: Peiran WANG","作者：王沛冉")

p_names = (("Introduction","Monitoring parameter selection","Monitoring method selection"),
           ("Einführung","Auswahl der Überwachungsparameter","Auswahl der Überwachungsmethode"),
           ("决策支持系统介绍","最佳监测参数选择","最佳监测方法选择"))


########## 第一章

k1t = ('Decision support system for structural health monitoring','Entscheidungssystem  für Structural Health Monitoring','结构健康监测的决策支持系统')

k1_0 = ('The purpose of this decision support system is to help users select the appropriate structural health monitoring strategy for hybrid structures in use phase.','Dieses Decision-Support-System soll den Nutzern helfen, die geeignete Strategie zur Überwachungssysteme für hybride Strukturen in der Nutzungsphase auszuwählen.','本决策支持系统的目的是帮助用户为使用阶段的混合结构选择合适的结构健康监测策略。')

k1_1 = ('The decision process is conducted in two steps. The first step is the selection of the monitoring parameters using the Analytic Hierarchy Process (AHP) decision method. The second step is the selection of the monitoring method using the Rule based reasoning (RBR) decision method. The following figure shows the framework of the Decision Support System.'
,
'Der Entscheidungsprozess wird in zwei Schritten durchgeführt. Der erste Schritt ist die Auswahl der Überwachungsparameter unter Verwendung der Analytic Hierarchy Process (AHP) Entscheidungsmethode. Der zweite Schritt ist die Auswahl der Überwachungsmethode, wobei die Rule based reasoning (RBR) Entscheidungsmethode verwendet wird. Das folgende Bild zeigt den Rahmen des Decision Support System.'
,
"决策过程分两步进行。第一步是使用层次分析法(AHP)决策方法选择监测参数。第二步是使用基于规则推理(RBR)的决策方法选择监测方法。下图显示了决策支持系统的框架。")


k1_2 = ('In the first step, a hierarchical model is created first. The monitoring parameters of the monitoring structure are used as decision objects. Parameters Sensitivity, Damage correlation, Monitoring Economy are used as indicators for the criteria and finally weights are calculated using the Analytic Hierarchy Process (AHP) decision method to determine the most appropriate monitoring parameters and recommend them to the user.'
,
'In ersten Schritt wird zuerst ein hierarchisches Modell erstellt. Die Überwachungsparameter der Überwachungsstruktur werden als Entscheidungsobjekte verwendet. Parameter Sensitivity, Damage correlation, Monitoring Economy werden als Indikatoren für die Kriterien verwendet und schließlich werden mithilfe der Analytic Hierarchy Process (AHP) Entscheidungsmethode Gewichtungen berechnet, um die am besten geeigneten Überwachungsparameter zu ermitteln und dem Benutzer zu empfehlen. '
,
'第一步，首先创建一个分层模型。监测结构的监测参数被作为决策对象。参数敏感性、损害相关性、监测经济性被用作标准的指标，最后，使用层次分析法（AHP）决策方法，计算权重以确定并向用户推荐最合适的监测参数。')

k1_3 = ('In the second step, the knowledge base (selection matrix) is compiled by analyzing the different monitoring methods based on the different attribute labels of the monitoring methods. Based on the information and desires of the decision maker, an "If-Then" statement is used to determine the best monitoring method.'
,
'Im zweiten Schritt wird die Wissensbasis (Auswahlmatrix) zusammengestellt, indem die verschiedenen Überwachungsmethoden anhand der verschiedenen Attributkennzeichnungen der Überwachungsmethoden analysiert werden. Ausgehend von den Informationen und Wünschen des Entscheidungsträgers wird eine "If-Then"-Anweisung verwendet, um die beste Überwachungsmethode zu ermitteln.',
'第二步，根据监测方法的不同属性标签，通过分析不同的监测方法，编制知识库（选择矩阵）。根据决策者的信息和意愿，采用 "If-Then "的语法来确定最佳的监测方法。')

##########   第二章 监测参数选择


k2t = ('Optimal monitoring parameter selection','Optimale Auswahl der Überwachungsparameter','最佳监测参数选择')

k2_1 = (
'The Analytic Hierarchy Process (AHP) methode was used to select appropriate monitoring parameters with three criteria, namely **Parameter Sensitivity**, **Damage Correlation**, and **Monitoring Economy**, followed by a comparison of the weights of each two.'
,
'Der analytische Hierarchieprozess (AHP) methode wurde verwendet, um geeignete Überwachungsparameter auszuwählen, wobei drei Kriterien, nämlich **Parameter Sensitivity**, **Damage Correlation**, und **Monitoring Economy**, berücksichtigt wurden, gefolgt von einem Vergleich der Gewichte der beiden Elemente.'
,
"通过分析层次过程（AHP）方法选择合适的监测参数，列出三个标准，即参数敏感性、损害相关性和监测经济性，然后是每两个的权重的比较。"
)

k2_2 = ("Monitoring parameters selection criteria","Kriterien für die Auswahl der Überwachungsparameter","监测参数选择标准"
)


k2_3_1 = ("__*Parameter Sensitivity*__: Reflects the sensitivity of the monitoring parameters to the degradation process of the system health condition, the more sensitive the parameters are, the easier it is to capture the performance degradation state of the structure"
,
"__*Parameter Sensitivity*__: Spiegelt die Empfindlichkeit der Überwachungsparameter für den Verschlechterungsprozess des Systemzustands wider; je empfindlicher die Parameter sind, desto leichter ist es, den Zustand der Leistungsverschlechterung der Struktur zu erfassen."
,
"参数敏感性：反映监测参数对系统健康状况退化过程的敏感性，参数越敏感，越容易捕捉到结构的性能退化状态"
)

k2_3_2 = ("__*Damage correlation*__: Monitoring parameters can reflect multiple health state performance of the structure (reflecting the type of damage)."
,
"__*Damage correlation*__: Die Überwachungsparameter können mehrere Zustandsmerkmale des Bauwerks wiedergeben (Art des Schadens)"
,
"损伤相关性：监测参数可以反映结构的几种状态特征（损坏类型）"
)

k2_3_3 = ("__*Monitoring economics*__: The cost price to be paid for monitoring, including monitoring cost, information transmission and processing cost, operation cost and labor cost, etc."
,
"__*Monitoring economics*__: Die für die Überwachung zu zahlenden Kosten, einschließlich der Kosten für die Überwachung, die Informationsübertragung und -verarbeitung, die Betriebskosten und die Arbeitskosten usw."
,
"监测经济性： 为监测而支付的费用，包括监测、信息传输和处理的费用、运营费用和人工费用等。"
)

k2_4 = ("Parameter Criterion Importance Assessment","Bewertung der Wichtigkeit der Parameterkriterien","参数标准重要性评估")

k2_5 = ("Please follow the 1-9  importance scale table in the figure to assess the importance of the three criteria of the monitored parameters according to your needs."
,
"Bitte folgen Sie der nachstehenden Wichtigkeitsskala von 1-9 Tabelle, um die Wichtigkeit der drei Kriterien für die Überwachungsparameter nach Ihren Bedürfnissen zu bewerten."
,
"请按照下图中9级重要性衡量表，根据您的需要，对监测参数的三个标准进行重要性评估。")

k2_6 = ("For monitoring parameters, which monitoring parameter criterion is more important. Please make a choice"
,"Welches Kriterium ist Ihrer Meinung nach für die Überwachungsparameter wichtiger? Bitte treffen Sie Ihre Wahl"
,"您觉得，哪种标准对于监测参数更重要？请您做出选择")


k2_7 = ("Click to determine optimal monitoring parameters","Klicken Sie hier, um optimale Überwachungsparameter zu bestimmen","点击进行最佳监测参数决策"  )


##################  第三章


k3t = ('Optimal monitoring method selection','Optimale Auswahl der Überwachungsmethode','最佳监测方法选择')

k3_1 = ("The second step is the selection of the monitoring method, using the Rule based reasoning (RBR) decision method. First, The knowledge base (selection matrix) is compiled by analyzing the different monitoring methods based on the different attribute labels of the monitoring methods. Based on the information and desires of the decision maker, an 'If-Then' statement is used to determine the best monitoring method.",
"Der zweite Schritt ist die Auswahl der Überwachungsmethode, wobei die Rule based reasoning (RBR) Entscheidungsmethode verwendet wird. Zuerst wird Die Wissensbasis (Auswahlmatrix) zusammengestellt, indem die verschiedenen Überwachungsmethoden anhand der verschiedenen Attributkennzeichnungen der Überwachungsmethoden analysiert werden. Ausgehend von den Informationen und Wünschen des Entscheidungsträgers wird eine 'If-Then'-Anweisung verwendet, um die beste Überwachungsmethode zu ermitteln.",
"第二步是选择监测方法，使用基于规则的推理（RBR）决策方法。首先，根据监测方法的不同属性标签，通过分析不同的监测方法，编制《知识库》（选择矩阵）。根据决策者的信息和愿望，使用 'If-Then'语句来确定最佳监测方法。"
)


k3_2 = ('Alternative monitoring methods','Alternative Überwachungsmethoden','备选监测方法介绍')

k3_21 = ("In the following tables, the alternative monitoring methods in the knowledge base are presented, along with their strengths and weaknesses.","In den folgenden Tabellen werden die alternativen Überwachungsmethoden in der Wissensbasis zusammen mit ihren Stärken und Schwächen vorgestellt.","以下表格中，介绍了知识库中的备选监测方法，以及他们的优点和不足")



k3_3 = ('Selection of attributes for monitoring methods','Auswahl von Eigenschaften für Überwachungsmethoden','监测方法的属性选择')

k3_31 = ("Please select the attributes related to the monitoring method according to the actual situation and wishes","Bitte wählen Sie die Attribute für die Überwachungsmethode entsprechend der tatsächlichen Situation und den Wünschen aus", "请根据实际情况和意愿，选择监测方法相关的属性")


k3_4 =("Recommended order of monitoring methods","Empfohlene Reihenfolge der Überwachungsmethoden","决策出监测方法的优先度排序")


k3_5 = ("References of optimal monitoring method","Quellen für optimale Überwachungsmethode","最佳监测方法有关的参考资料")

k3_51 = ("Park, K., Torbol, M., & Kim, S. (2018). Vision-based natural frequency identification using laser speckle imaging and parallel computing. _Computer-Aided Civil and Infrastructure Engineering_","Rucevskis, S., Janeliukstis, R., Akishin, P., & Chate, A. (2016). Mode shape-based damage detection in plate structure without baseline data. _Structural Control and Health Monitoring_","Medeiros, R., Souza, G., Marques, D., Flor, F., & Tita, V. (2021). Vibration-based structural monitoring of bi-clamped metal-composite bonded joint: Experimental and numerical analyses. _The Journal of Adhesion_","Nauman, S.; Cristian, I.; Koncar, V. (2011): Simultaneous application of fibrous piezoresistive sensors for compression and traction detection in glass laminate composites. _Sensors_","Du, F., Wang, G., Weng, J., Fan, H., & Xu, C. (2022). High-precision probabilistic imaging for interface debonding monitoring based on electromechanical impedance. _AIAA Journal_","Zhao, X., Royer, R. L., Owens, S. E., & Rose, J. L. (2011). Ultrasonic lamb wave tomography in structural health monitoring. _Smart Materials and Structures_","Meruane, V., Aichele, D., Ruiz, R., & López Droguett, E. (2021). A deep learning framework for damage assessment of composite sandwich structures. _Shock and Vibration, 2021_")

k3_52 = ("Pan, J., Zhang, Z., Wu, J., Ramakrishnan, K. R., & Singh, H. K. (2019). A novel method of vibration modes selection for improving accuracy of frequency-based damage detection._Composites Part B: Engineering_","Govindasamy, M., Kamalakannan, G., Kesavan, C., & Meenashisundaram, G. K. (2020). Damage detection in glass/epoxy laminated composite plates using modal curvature for structural health monitoring applications. _ournal of Composites Science_","Nayyar, A., Baneen, U., Ahsan, M., Zilqurnain Naqvi, S. A., & Israr, A. (2022). Damage detection based on output-only measurements using cepstrum analysis and a baseline-free frequency response function curvature method. _Science Progress_","Wan, H. P., & Ni, Y. Q. (2018). Bayesian modeling approach for forecast of structural stress response using structural health monitoring data. _Journal of Structural Engineering_","Pravallika, N. N. S., Reddy, V. M., & Reddy, B. S. K. (2021). Impedance based service life assessment of corroded structures with cross correlation analysis. _E3S Web of Conferences_","Wang, S.,Wu,W., Shen, Y., Liu, Y., & Jiang, S. (2020a). Influence of the pzt sensor array configuration on lamb wave tomography imaging with the rapid algorithm for hole and crack detection. _Sensors_","Zenzen, R., Khatir, S., Belaidi, I., Le Thanh, C., &Wahab, M. A. (2020). A modified transmissibility indicator and artificial neural network for damage identification and quantification in laminated composite structures. _Composite Structures_")


k3_6 = ("Click to determine optimal monitoring method","Klicken Sie hier, um optimale Überwachungsmethode zu bestimmen","点击进行最佳监测方法决策"  )
