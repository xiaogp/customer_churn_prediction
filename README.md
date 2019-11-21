# customer_churn_prediction
零售用户流失二分类预测，数据清洗使用sparksql，FM模型训练使用tensorflow，接口服务使用tensorflow_model_server，GBDT模型训练使用XGBOOST，接口服务使用flask

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


FM因子分解机
数据格式转换 libsvm

```
python libsvm_transform.py 
```

```
0 1:1 7:1 13:1 20:1 27:1 33:1 42:1 55:1 61:1 67:1 74:1 79:1 86:1 91:1 97:1 103:1 109:1 115:1 119:1 125:1 131:1 137:1 142:1 148:1 151:1 154:1 157:1 160:1 163:1 166:1 169:1 172:1 175:1 178:1 181:1 184:1
1 1:1 11:1 15:1 24:1 29:1 37:1 39:1 59:1 62:1 68:1 76:1 81:1 87:1 91:1 97:1 103:1 109:1 115:1 122:1 125:1 135:1 137:1 146:1 148:1 152:1 154:1 157:1 160:1 163:1 166:1 169:1 172:1 175:1 178:1 181:1 184:1
1 5:1 8:1 16:1 23:1 31:1 37:1 39:1 59:1 62:1 70:1 76:1 81:1 86:1 94:1 98:1 107:1 111:1 115:1 120:1 125:1 132:1 137:1 144:1 149:1 151:1 154:1 157:1 160:1 164:1 166:1 169:1 173:1 175:1 179:1 181:1 184:1
```

fm_csv将libsvm数据解析成csv，serving格式也是csv

churn_featindex.txt对照表

```
0:other 0
0:30以下 1
0:30-60 2
0:60-90 3
0:90-120 4
0:120以上 5
1:other 6
1:30以下 7
1:30-60 8
1:60-90 9
1:90-120 10
1:120以上 11
```

模型训练
```

python fm_churn_csv.py --feature_size 186
```

fm_libsvm通过sparse_placeholder直接训练libsvm格式数据

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

accuracy: 0.7592295588733791
precision: 0.7635289710090631
reall: 0.8423797379298215
f1: 0.8010185522008003
auc: 0.8263173355592242
```

rest接口测试
启动服务
```
docker run -t --rm -p 8501:8501 -v "/******/******/churn/FM_churn.pb:/models/FM/" -e MODEL_NAME=FM tensorflow/serving

```

接口测试
```
curl -d '{"instances": [{"input_x": [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0,0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0,0,0,0,1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0,1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0,0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0,0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0,0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0,0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0,0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1,0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]}], "signature_name":"my_signature"}' -X POST http://localhost:8501/v1/models/FM:predict
{
    "predictions": [0.472961
    ]
```


GBDT梯度提升树
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

启动接口服务
```
python churn_xgb_server.py
```

接口测试

![](/GBDT/img/server_test.png)




sparkML的模型结果
```
AreaUnderROC: 0.831
accuracy: 0.765
precision: 0.766
recall: 0.850
fMeasure: 0.806
```
