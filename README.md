# customer_churn_prediction
零售电商客户流失模型，基于tensorflow，xgboost4j-spark实现线性模型LR，FM，GBDT，RF，进行模型效果对比，离线/在线serving部署方式总结。

## 模型的部署方式
- LR使用LibSVM格式的数据集， 采用 TFRecords + tf.data.Dataset + model + tf_model_server的tensorflow编程模型。

- FM分别使用了csv和LibSVM两种格式的数据，采用 tf.placeholder / tf.Sparse_placeholder+ model + tf_model_server的tensorflow编程模型。

- GBDT使用csv格式数据，采用sklearn的自定义Pipeline配合xgboost的sklearn接口整体封装特征工程和模型为一个完整的pipeline的pkl序列化文件，再包上Flask的API模型接口。GBDT也采用xgboost4j-spark进行模型效果对比。

- RF采用SparkSQL的原始数据，采用Spark ML组件，配合airflow+spark submit定时任务部署。

## 模型对比
| 指标/模型  | LR  | FM  | GBDT  | GBDT  |RF|
| ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |
| 框架  | tensorflow  | tensorflow  | xgboost  |  xgboost4j-spark |SparkML|
| accuracy  | 0.749  | 0.759  | **0.766**  | 0.763  |0.765|
| precision  | 0.750  | 0.764  | 0.765  | **0.766**  |**0.766**|
| reccall  | 0.845  | 0.842  | **0.853**  | 0.847  |0.850 |
| auc_score  | 0.816 |  0.826  | **0.833**  | 0.832  |0.831|
| f1_score  | 0.795  | 0.801  | **0.807**  |  0.805 |0.806|

## 特征说明
特征类型

| 特征  | 备注  | 特征  | 备注  |
| ------------ | ------------ | ------------ | ------------ |
| shop_duration  | 购物时间跨度  | recent  | 6个月R值  |
| monetary  | 6个月M值  | max_amount  | 6个月最大一次购物金额  |
| items_count  | 总购买商品数  | valid_points_sum  | 有效积分数  |
| CHANNEL_NUM_ID  | 注册渠道  | member_day  | 会员年限  |
| VIP_TYPE_NUM_ID  | 会员卡等级  | frequence  | 6个月F值  |
| avg_amount  | 客单价  | item_count_turn  | 单次购买商品数  |
| avg_piece_amount  | 单品购买价格  | monetary3  | 3个月M值  |
| max_amount3  | 3个月最大一次购物金额  | items_count3  | 3个月购买总商品数  |
| frequence3  | 3个月F值  | shops_count  | 跨门店购买数  |
| promote_percent  | 促销购买比例  | wxapp_diff  | 微信小程序购买R值  |
| store_diff  | 门店购买R值  | shop_channel  | 购物渠道  |
| week_percent  | 周末购物比例  | infant_group  | 母婴客群  |
| water_product_group  | 水产客群  | meat_group  | 肉禽客群  |
| beauty_group  | 美妆客群  | health_group  | 保健客群  |
| fruits_group  | 水果客群  | vegetables_group  | 蔬菜客群  |
| pets_group  | 家有宠物  | snacks_group  | 零食客群  |
| smoke_group  | 烟民  | milk_group  | 奶制品客群  |
| instant_group  | 方便食品客群  | grain_group  | 粮油食品客群  |

## 数据预览
数据位置/LR/data/churn_train_sample.csv，展示表头和第一行数据
```
head -2 churn_train_sample.csv
USR_NUM_ID,shop_duration,recent,monetary,max_amount,items_count,valid_points_sum,CHANNEL_NUM_ID,member_day,VIP_TYPE_NUM_ID,frequence,avg_amount,item_count_turn,avg_piece_amount,monetary3,max_amount3,items_count3,frequence3,shops_count,promote_percent,wxapp_diff,store_diff,shop_channel,week_percent,infant_group,water_product_group,meat_group,beauty_group,health_group,fruits_group,vegetables_group,pets_group,snacks_group,smoke_group,milk_group,instant_group,grain_group,label
464087,30以下,30以下,100以下,20-50,1-5,50-100,7,30以下,0,1以下,50-100,2-5,10-20,50-100,20-50,1-5,1以下,1以下,0.2-0.4,30以下,30以下,unknow,0.8以上,unknow,unknow,unknow,美妆客
```

csv转LibSVM格式 ，脚本位置/FM/fm_libsvm/libsvm_transform.py
查看LibSVM的对照表/FM/fm_libsvm/libsvm_transform.py
```
head -5 churn_featindex.txt
0:other 0
0:30以下 1
0:30-60 2
0:60-90 3
0:90-120 4
```
执行转化脚本
```
python libsvm_transform.py
```
LibSVM数据预览
```
head -2 churn_train_sample.svm
1 1:1 7:1 13:1 21:1 28:1 34:1 42:1 55:1 61:1 67:1 76:1 81:1 86:1 93:1 98:1 104:1 109:1 115:1 120:1 125:1 131:1 137:1 146:1 148:1 151:1 154:1 158:1 160:1 163:1 166:1 169:1 172:1 175:1 178:1 181:1 184:1
0 5:1 7:1 15:1 22:1 31:1 36:1 39:1 59:1 62:1 69:1 76:1 81:1 86:1 94:1 99:1 106:1 110:1 115:1 121:1 125:1 131:1 137:1 143:1 148:1 151:1 154:1 157:1 160:1 164:1 166:1 169:1 173:1 175:1 179:1 182:1 185:1
```

## LR逻辑回归
将LIbSVM数据制作成TFRecords数据
```
python TFRecord_process.py
```
训练模型
```
python main.py
```
模型训练过程
```
step: 9100 loss: 0.52239525 auc: 0.81408113
step: 9200 loss: 0.50950295 auc: 0.81406915
step: 9300 loss: 0.5170015 auc: 0.8140943
step: 9400 loss: 0.5239074 auc: 0.8141037
step: 9500 loss: 0.504278 auc: 0.81413954
step: 9600 loss: 0.5412767 auc: 0.8141376
step: 9700 loss: 0.5137014 auc: 0.81412816
step: 9800 loss: 0.46152985 auc: 0.8141491
step: 9900 loss: 0.48090518 auc: 0.8141693
step: 10000 loss: 0.49998602 auc: 0.8141641
[evaluation] loss: 0.51270264 auc: 0.814165
```
测试集评价
```
accuracy: 0.7492069434817584
precision: 0.7503747423646243
reall: 0.8452554744525548
f1: 0.7949941686862588
auc: 0.8156375812964103
```

项目文件树结构
```
├── TFRecord_process.py
├── __pycache__
│   ├── model.cpython-37.pyc
│   ├── preprocessing.cpython-37.pyc
│   └── utils.cpython-37.pyc
├── churn_lr.pb
│   ├── 001
│   │   ├── saved_model.pb
│   │   └── variables
│   │       ├── variables.data-00000-of-00001
│   │       └── variables.index
│   └── models
├── config.yml
├── data
│   ├── churn_featindex.txt
│   ├── churn_test.svm
│   ├── churn_train.svm
│   ├── test.tfrecords
│   └── train.tfrecords
├── main.py
├── model.py
└── utils.py
```

使用docker的tensorflow_model_server镜像部署模型，rest接口测试启动服务
```
docker run --rm -d -p 8501:8501 -v "/****/customer_churn_prediction/LR/churn_lr.pb:/models/churn_lr/" -e 	MODEL_NAME=churn_lr tensorflow/serving
```

接口测试
```
curl -d '{"instances": [{"input_x": [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0,0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0,0,0,0,1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0,1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0,0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0,0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0,0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0,0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0,0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1,0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]}], "signature_name":"my_signature"}' -X POST http://localhost:8501/v1/models/churn_lr:predict
{
    "predictions": [0.497120261
    ]
```


## FM 因子分解机
 fm_churn_csv.py采用csv个数数据训练模型
```
python fm_churn_csv.py --feature_size 186
```
fm_libsvm.py采用sparse_placeholder直接训练libsvm格式数据

```
python fm_churn_libsvm.py 
```

模型训练过程
```
step: 76100 loss: 0.5005622 auc: 0.82709
step: 76200 loss: 0.50755 auc: 0.8270913
step: 76300 loss: 0.48795617 auc: 0.8270925
step: 76400 loss: 0.5073022 auc: 0.8270925
step: 76500 loss: 0.5022451 auc: 0.8270947
step: 76600 loss: 0.5266277 auc: 0.8270936
step: 76700 loss: 0.50896007 auc: 0.8270941
step: 76800 loss: 0.46825206 auc: 0.8270943
step: 76900 loss: 0.49328235 auc: 0.8270949
step: 77000 loss: 0.5090138 auc: 0.82709527
[evaluation] loss 0.4988083 auc: 0.82709527 
```
测试集评价
```
accuracy: 0.7592295588733791
precision: 0.7635289710090631
reall: 0.8423797379298215
f1: 0.8010185522008003
auc: 0.8263173355592242
```

使用docker的tensorflow_model_server镜像部署模型，rest接口测试启动服务
```
docker run -t --rm -p 8501:8501 -v "/****/customer_churn_prediction/FM/fm_csv/FM_churn.pb:/models/FM/" -e MODEL_NAME=FM tensorflow/serving

```
接口测试
```
curl -d '{"instances": [{"input_x": [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0,0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0,0,0,0,1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0,1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0,0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0,0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0,0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0,0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0,0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1,0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]}], "signature_name":"my_signature"}' -X POST http://localhost:8501/v1/models/FM:predict
{
    "predictions": [0.472961
    ]
```

## GBDT梯度提升树
模型训练
```
python churn_xgb.py
```

GBDT测试集模型结果
```
acc: 0.7656144859931294
pri: 0.7654276063379557
rec: 0.8530070349277994
auc: 0.8327608699836433
```

启动flask web server
```
python churn_xgb_server.py
```
postman接口测试

![](/GBDT/img/server_test.png)

## xgboost4j-spark
提交spark任务
```
spark-submit --master local[*] --class com.mycom.myproject.churn_xgb4j_spark myproject-1.0-SNAPSHOT.jar
```
```   
accuracy: 0.763                                                                  
precision: 0.766                                                                 
recall: 0.847      
fMeasure: 0.805      
AreaUnderROC: 0.832     
```

## RF 随机森林  
```
spark-submit --master local[*] --class com.mycom.myproject.randomforest_churn myproject-1.0-SNAPSHOT.jar
```
```
AreaUnderROC: 0.831
accuracy: 0.765
precision: 0.766
recall: 0.850
fMeasure: 0.806
```
