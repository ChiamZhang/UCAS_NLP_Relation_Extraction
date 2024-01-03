# UCAS_NLP_Relation_Extraction
国科大NLP SemEval-2010 Task8 关系抽取作业
**SemEval-2010 Task8 关系抽取作业**

1. 实验目的
   1. 进一步加深对信息抽取章节中关系抽取任务基本目标和流程的理解。
   2. 掌握卷积神经网络、循环神经网络等处理文本的各项技术。
   3. 加强对pytorch、tensorflow等深度学习框架的使用能力。
2. 实验要求
   1. 任选一个深度学习框架建立一个关系抽取模型，实现在英文数据集SemEval-2010 Task8上进行的关系抽取。
   2. 按规定时间在课程网站提交实验报告、代码。
3. 数据集介绍

本次实验所用的数据集是完全监督的关系抽取数据集SemEval-2010 Task8，其全部数据均在附件的SemEval2010_task8_all_data.zip压缩包中。这个数据集包含了10717个样本，其中，8000个为训练样例，2717个测试样例。其中，训练集数据在TRAIN_FILE.TXT，测试集数据在TEST_FILE.TXT中。
这个数据集包含了9种基本关系和1种人工分类关系Other。这九种关系分别为:Cause-Effect、Component-Whole、Entity-Destination、Product-Producer、Entity-Origin、Member-Collection、Message-Topic、Content-Container、Instrument-Agency。特别需要说明的是，另一种关系Other是一种人工分类关系，其表示的是不属于这9种关系之一，而并不是一种独特的关系。
在SemEval-2010 Task8的训练集以及测试集中，上述10种关系的数目和比例如下所示：

| **类别** | **训练集中的数目(比例)** | **测试集中的数目(比例)** |
| --- | --- | --- |
| **Other** | 1410 (17.63%) | 454 (16.71%) |
| **Cause-Effect** | 1003 (12.54%) | 328 (12.07%) |
| **Component-Whole** | 941 (11.76%) | 312 (11.48%) |
| **Entity-Destination** | 845 (10.56%) | 292 (10.75%) |
| **Product-Producer** | 717 ( 8.96%) | 261 ( 9.61%) |
| **Entity-Origin** | 716 ( 8.95%) | 258 ( 9.50%) |
| **Member-Collection** | 690 ( 8.63%) | 233 ( 8.58%) |
| **Message-Topic** | 634 ( 7.92%) | 231 ( 8.50%) |
| **Content-Container** | 540 ( 6.75%) | 192 ( 7.07%) |
| **Instrument-Agency** | 504 ( 6.30%) | 156 ( 5.74%) |


SemEval-2010 Task8的训练集和测试集的数据格式相同。每一个样例由三行内容组成，其中第一行为样例的序号以及具体的语句内容，第二行为此语句中两个实体的关系，第三行为对于此样例的注释：
8001  " The most common <e1>audits</e1> were about <e2>waste</e2> and recycling . " 
Message-Topic(e1 ,e2)
Comment: Assuming an audit = an audit document .
在样例的每一个句子中，对于实体，其使用了<e1></e1><e2></e2>标签将其标注出来。在说明关系的时候，考虑了这两个实体的顺序。这意味着Cause-Effect(e1,e2)和Cause-Effect(e2,e1)是两个不同的关系。因此，在实际实验中，通常认为在此数据集中包含了19种关系。

1. 参考模型

本次实验要求实现一种关系抽取模型即可，可采用**下面三篇参考论文中的任意一种或自己设计的结构**均可。

   1. 使用CNN进行关系抽取，可参考2015年ACL的基于Ranking Loss进行关系抽取的论文[[[1]](#endnote-2)] ，要求实现其中的CNN和Ranking Loss。
   2. 使用RNN/LSTM进行关系抽取，可参考2016年ACL的基于LSTM+Attention的关系抽取论文[[[2]](#endnote-3)]，要求实现其结构。
   3. 计算资源比较充足的同学，也可采用BERT进行关系抽取，可参考2019年ACL的基于BERT的关系抽取论文[[[3]](#endnote-4)]，要求实现其结构。

1. 作业提交说明
   1. 要求使用上述参考论文中的结构或自己实现的结构来实现关系抽取模型。
   2. 提交的作业除代码外，需包含一份简要报告来说明自己使用的模型结构以及达到的效果。如果有自己的一些独特的思考和实现可以详细写出。
   3. 测评指标使用的是官方给出的scorer，其实现基于perl语言，位于

SemEval2010_task8_all_data\SemEval2010_task8_scorer-v1.2路径下，
为semeval2010_task8_scorer-v1.2.pl，实验报告中需给出测评得到的macro-F1 score。(也可自行计算F1，但推荐使用官方scorer)

Scorer的使用方式在README.txt中进行了详细介绍，使用的测评指标为输出文件的最后一行，即The official score is (9+1)-way evaluation with directionality taken into account: macro-averaged F1后面的数值。
如使用ubuntu环境运行，则scorer的使用方式可以参考如下代码。如使用windows环境运行，则需要配置perl环境或自行计算F1值。其中perl_path为scorer路径。而output_predict_file为模型预测文件的路径，target_path为标注test的label文件路径，其格式均为: ![](https://cdn.nlark.com/yuque/0/2024/png/40576197/1704245623117-3ea7a920-3280-4c0b-88e7-043879612116.png#averageHue=%23302e2d&id=MPrxX&originHeight=48&originWidth=259&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=)，即序号加指标符再加预测的标签：

perl_path = os.path.join(os.path.curdir,
                                   "SemEval2010_task8_scorer-v1.2",
                                   "semeval2010_task8_scorer-v1.2.pl")
          target_path = os.path.join(os.path.curdir, "resource", "target.txt")
          process = subprocess.Popen(["perl", perl_path, output_predict_file, target_path], stdout=subprocess.PIPE)
          str_parse = str(process.communicate()[0]).split("\\n")[-2]
          idx = str_parse.find('%')
          f1_score = float(str_parse[idx - 5:idx])

1. 参考文献
2. [] Santos, Cicero Nogueira dos, Bing Xiang, and Bowen Zhou. "Classifying relations by ranking with convolutional neural networks." arXiv preprint arXiv:1504.06580 (2015). [↑](#endnote-ref-2)
3. [] Zhou, Peng, et al. "Attention-based bidirectional long short-term memory networks for relation classification." Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers). 2016. [↑](#endnote-ref-3)
4. [] Soares, Livio Baldini, et al. "Matching the Blanks: Distributional Similarity for Relation Learning." Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics. 2019. [↑](#endnote-ref-4)
