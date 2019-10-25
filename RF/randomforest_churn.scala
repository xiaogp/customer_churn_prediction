import utils.getDiffDatetime
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.expressions.Window
import utils.{AUCEvaluation, PREvaluation, getProPerties, getProba}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.feature.{OneHotEncoderEstimator, StringIndexer, VectorAssembler}


object ChurnModel {
  def main(args: Array[String]): Unit = {
    Logger.getRootLogger.setLevel(Level.WARN)
    val properties = getProPerties(args(0))
    val saveTable = properties.getProperty("ChurnModelSaveTable")
    val saveHdfs = properties.getProperty("ChurnModelHDFSPath")
    val spark = SparkSession.builder().appName("ChurnModel").master("yarn").getOrCreate()

    addLabel(spark, "2019-07-01", "2019-09-30", saveTable)
    val modelDF = spark.sql(s"select * from ${saveTable} order by rand()")

    // 样本划分
    val Array(trainDF, testDF) = modelDF.randomSplit(Array(0.8, 0.2))

    // 训练模型保存模型文件
    modelPipeline(trainDF, saveHdfs)

    // 测试集评价
    val ppl = PipelineModel.load(saveHdfs)
    val testPred = ppl.transform(testDF).cache()
    AUCEvaluation(testPred, labelCol = "label", predCol = "rawPrediction")
    PREvaluation(testPred, labelCol = "label", predCol = "prediction")

    spark.close()

  }


  def churnPreprocessing(spark: SparkSession, startDate: String, endDate: String): DataFrame = {
    import spark.implicits._

    val staticData = spark.sql(
      s"select USR_NUM_ID, CHANNEL_NUM_ID, datediff('${endDate}', REGISTRAT_DATE) as member_day, VIP_TYPE_NUM_ID" +
        s" from members_data where to_date(date_rank) >= '${startDate}' and to_date(date_rank) <= '${endDate}'")

    val salesData = spark.sql(
      s"select a.ORDER_DATE, a.TML_NUM_ID, a.SUB_UNIT_NUM_ID, a.USR_NUM_ID, a.TRADE_AMOUNT, a.ITEM_NUM_ID, a.DISCOUNT_AMOUNT, " +
        s"b.PTY_NUM_1 from sales_data a left join goods_data b " +
        s"on a.ITEM_NUM_ID = b.ITEM_NUM_ID where to_date(a.ORDER_DATE) >= '${startDate}' and to_date(a.ORDER_DATE) <= '${endDate}' and " +
        s"to_date(b.rank_date) >= '${startDate}' and to_date(b.rank_date) <= '${endDate}'").cache()

    val pointsData = spark.sql(
      "select USR_NUM_ID, RESERVED_INTEGRAL from points_data " +
        s"where to_date(TSC_DTME) >= '${startDate}' and to_date(TSC_DTME) <= '${endDate}'")

    // 提取有效积分总数
    val validPointsData = pointsData.groupBy($"USR_NUM_ID").agg(sum($"RESERVED_INTEGRAL").alias("valid_points_sum"))

    // 购物行为特征
    val salesDataWithColumns = salesData
      .withColumn("last_shop_date", max($"ORDER_DATE").over(Window.partitionBy($"USR_NUM_ID")).cast("date"))
      .withColumn("earliest_shop_date", min($"ORDER_DATE").over(Window.partitionBy($"USR_NUM_ID")).cast("date"))
      .withColumn("shop_duration", datediff($"last_shop_date", $"earliest_shop_date"))
      .withColumn("now_date", lit(endDate).cast("date"))
      .withColumn("recent", datediff($"now_date", $"last_shop_date"))
      .withColumn("monetary", round(sum($"TRADE_AMOUNT").over(Window.partitionBy($"USR_NUM_ID")), 2))
      .withColumn("max_amount", max($"TRADE_AMOUNT").over(Window.partitionBy($"USR_NUM_ID")))
      .withColumn("save_amount", sum($"DISCOUNT_AMOUNT").over(Window.partitionBy($"USR_NUM_ID")))
      .withColumn("items_count", count($"ITEM_NUM_ID").over(Window.partitionBy($"USR_NUM_ID")))
      .select($"USR_NUM_ID", $"shop_duration", $"recent", $"monetary", $"max_amount", $"save_amount", $"items_count")
      .distinct()

    val frequenceData = salesData
      .groupBy("USR_NUM_ID")
      .agg(countDistinct($"TML_NUM_ID").alias("frequence"))

    val crossShopData = salesData
      .groupBy("USR_NUM_ID")
      .agg(countDistinct($"SUB_UNIT_NUM_ID").alias("shops_count"))

    val promoteData = salesData
      .filter(col("PRICE_TYPE") =!= 0)
      .groupBy(col("USR_NUM_ID")).count()
      .withColumnRenamed("count", "promote_items_count")

    def repurchase_calulate(endDate: String, window_length: Int, label: Int): DataFrame = {
      val repurchaseData = salesData
        .filter($"ORDER_DATE" >= getDiffDatetime(window_length, endDate))
        .groupBy($"USR_NUM_ID")
        .agg(countDistinct($"TML_NUM_ID").alias("count"))
        .filter($"count" > 1)
        .withColumn(s"last_${label}_repurchase", lit(label)).drop("count")
      repurchaseData
    }

    val last_1_repurchase = repurchase_calulate(endDate, 30, 1)
    val last_2_repurchase = repurchase_calulate(endDate, 90, 2)
    val last_4_repurchase = repurchase_calulate(endDate, 180, 4)

    // 小程序会员标识
    val wxappData = salesData.filter($"WX_APP_SIGN" === 1)
      .withColumn("last_wxapp_date", max($"ORDER_DATE").over(Window.partitionBy($"USR_NUM_ID")).cast("date"))
      .withColumn("now_date", lit(endDate).cast("date"))
      .withColumn("wxapp_diff", datediff($"now_date", $"last_wxapp_date"))
      .withColumn("wxapp_member", lit("小程序会员"))
      .select($"USR_NUM_ID", $"wxapp_diff", $"wxapp_member").distinct()

    // 门店会员标识
    val storeData = salesData.filter($"WX_APP_SIGN" =!= 1)
      .withColumn("last_store_date", max($"ORDER_DATE").over(Window.partitionBy($"USR_NUM_ID")).cast("date"))
      .withColumn("now_date", lit(endDate).cast("date"))
      .withColumn("store_diff", datediff($"now_date", $"last_store_date"))
      .withColumn("store_member", lit("门店会员"))
      .select($"USR_NUM_ID", $"store_diff", $"store_member").distinct()

    def shopchannelUdf = udf((x: String, y: String) => {
      if (x == "小程序会员" && y == "门店会员") Some("门店和小程序")
      else if (x == "小程序会员" && y != "门店会员") Some("仅小程序")
      else if (x != "小程序会员" && y != "门店会员") Some("仅门店")
      else None
    })

    val timePredilection = salesData
      .withColumn("is_weekend", udf((x: Int) => {
        if (List(1, 7).contains(x)) 1 else 0
      }).apply(dayofweek($"ORDER_DATE")))
      .select($"USR_NUM_ID", $"TML_NUM_ID", $"is_weekend").distinct()
      .groupBy("USR_NUM_ID")
      .agg(bround(mean($"is_weekend"), 2).alias("week_percent"))

    def groupFunc(groupCol: String, groupLabel: String, itemID: List[Int]): DataFrame = {
      val groupData = salesData
        .filter($"PTY_NUM_1".isin(itemID: _*))
        .select($"USR_NUM_ID").distinct()
        .withColumn(groupCol, lit(groupLabel))
      groupData
    }

    val muyinData = groupFunc("infant_group", "母婴客群", List(2501))
    val shuichanData = groupFunc("water_product_group", "水产客群", List(2104))
    val rouqinData = groupFunc("meat_group", "肉禽客群", List(2101, 2102))
    val meizhuangData = groupFunc("beauty_group", "美妆客群", List(2502))
    val baojianData = groupFunc("health_group", "保健客群", List(2307))
    val shuiguoData = groupFunc("fruits_group", "水果客群", List(2106))
    val shucaiData = groupFunc("vegetables_group", "蔬菜客群", List(2105))
    val petsData = groupFunc("pets_group", "家有宠物", List(2507))
    val snacksData = groupFunc("snacks_group", "零食客群", List(2309))
    val smokeData = groupFunc("smoke_group", "烟民", List(2403))
    val milkData = groupFunc("milk_group", "奶饮品客群", List(2306, 2407))
    val instantData = groupFunc("instant_group", "方便速食客群", List(2304))
    val grainData = groupFunc("grain_group", "粮油客群", List(2301))

    // 补充近3个月的 总购买金额 购买次数 购买商品数 最大一次消费金额
    val last3Date = getDiffDatetime(90, endDate)
    val salesDataLast3Month = spark.sql(
      s"select TML_NUM_ID, USR_NUM_ID, TRADE_AMOUNT, ITEM_NUM_ID from sales_data " +
        s"where to_date(ORDER_DATE) >= '${last3Date}' and to_date(ORDER_DATE) <= '${endDate}'")
    val salesDataLast3MonthWithColumns = salesDataLast3Month
      .withColumn("monetary3", round(sum($"TRADE_AMOUNT").over(Window.partitionBy($"USR_NUM_ID")), 2))
      .withColumn("max_amount3", max($"TRADE_AMOUNT").over(Window.partitionBy($"USR_NUM_ID")))
      .withColumn("items_count3", count($"ITEM_NUM_ID").over(Window.partitionBy($"USR_NUM_ID")))
      .select($"USR_NUM_ID", $"monetary3", $"max_amount3", $"items_count3")
      .distinct()
    val frequenceDataLast3 = salesDataLast3Month
      .groupBy("USR_NUM_ID")
      .agg(countDistinct($"TML_NUM_ID").alias("frequence3"))

    val featureData = salesDataWithColumns
      .join(validPointsData, Seq("USR_NUM_ID"), "left_outer")
      .join(staticData, Seq("USR_NUM_ID"), "left_outer")
      .join(frequenceData, Seq("USR_NUM_ID"), "left_outer")
      .withColumn("avg_amount", bround($"monetary" / $"frequence", 2))
      .withColumn("item_count_turn", bround($"items_count" / $"frequence", 2))
      .withColumn("avg_piece_amount", bround($"monetary" / $"items_count", 2))
      .join(salesDataLast3MonthWithColumns, Seq("USR_NUM_ID"), "left_outer")
      .join(frequenceDataLast3, Seq("USR_NUM_ID"), "left_outer")
      .join(crossShopData, Seq("USR_NUM_ID"), "left_outer")
      .join(promoteData, Seq("USR_NUM_ID"), "left_outer")
      .na.fill(0, cols = Array("promote_items_count"))
      .withColumn("promote_percent", round($"promote_items_count" / $"items_count", 2))
      .join(wxappData, Seq("USR_NUM_ID"), "left_outer")
      .join(storeData, Seq("USR_NUM_ID"), "left_outer")
      .withColumn("shop_channel", shopchannelUdf($"wxapp_member", $"store_member"))
      .join(timePredilection, Seq("USR_NUM_ID"), "left_outer")
      .join(muyinData, Seq("USR_NUM_ID"), "left_outer")
      .join(shuichanData, Seq("USR_NUM_ID"), "left_outer")
      .join(rouqinData, Seq("USR_NUM_ID"), "left_outer")
      .join(meizhuangData, Seq("USR_NUM_ID"), "left_outer")
      .join(baojianData, Seq("USR_NUM_ID"), "left_outer")
      .join(shuiguoData, Seq("USR_NUM_ID"), "left_outer")
      .join(shucaiData, Seq("USR_NUM_ID"), "left_outer")
      .join(petsData, Seq("USR_NUM_ID"), "left_outer")
      .join(snacksData, Seq("USR_NUM_ID"), "left_outer")
      .join(smokeData, Seq("USR_NUM_ID"), "left_outer")
      .join(milkData, Seq("USR_NUM_ID"), "left_outer")
      .join(instantData, Seq("USR_NUM_ID"), "left_outer")
      .join(grainData, Seq("USR_NUM_ID"), "left_outer")
      .drop("wxapp_member").drop("store_member").drop("promote_items_count")
      .na.fill("unknow").na.fill(0)

    featureData
  }

  /**
    * 补充标签列
    *
    * @param spark
    * @param startDate 开始时间窗口
    * @param endDate   结束时间窗口
    */
  def addLabel(spark: SparkSession, startDate: String, endDate: String, saveTable: String): Unit = {
    import spark.implicits._

    val featureData = churnPreprocessing(spark, "2019-01-01", "2019-06-30")
    // 正样本是在0-6月买并且在7-9月也购买的会员
    // 负样本是在0-6月买在7-9月没有购买的会员
    val buyedData = spark.sql(s"select distinct USR_NUM_ID, 0 as label from sales_data " +
      s"where to_date(ORDER_DATE) >= '${startDate}' and to_date(ORDER_DATE) <= '${endDate}'")
    val joinLabelData = featureData.join(buyedData, Seq("USR_NUM_ID"), "left_outer").na.fill(1, cols = Array("label"))
    val Array(negativeCount, positiveCount) = joinLabelData.groupBy($"label").count().sort("label").collect().map(x => x(1).asInstanceOf[Long])
    println(s"正样本: ${positiveCount}")
    println(s"负样本: ${negativeCount}")

    joinLabelData.write.format("orc").mode("overwrite").saveAsTable(saveTable)

  }

  /**
    * 特征工程和模型pipeline
    *
    * @param trainDF
    */
  def modelPipeline(trainDF: DataFrame, saveHdfs: String): Unit = {
    trainDF.cache()
    val continueCols = Array("shop_duration", "recent", "monetary", "max_amount", "save_amount", "items_count", "valid_points_sum",
      "member_day", "frequence", "avg_amount", "item_count_turn", "avg_piece_amount", "shops_count", "promote_percent",
      "wxapp_diff", "store_diff", "week_percent", "monetary3", "max_amount3", "items_count3", "frequence3")
    val categoryCols = Array("CHANNEL_NUM_ID", "VIP_TYPE_NUM_ID", "shop_channel", "infant_group", "water_product_group", "meat_group",
      "beauty_group", "health_group", "fruits_group", "vegetables_group", "pets_group", "snacks_group",
      "smoke_group", "milk_group", "instant_group", "grain_group")

    val stringTransform = categoryCols.map(
      name => new StringIndexer()
        .setInputCol(name)
        .setOutputCol(s"${name}_index")
        .setHandleInvalid("skip")
    )

    val oneHotTransform = new OneHotEncoderEstimator()
      .setDropLast(false) // 保留所有分类
      .setInputCols(categoryCols map (name => s"${name}_index"))
      .setOutputCols(categoryCols map (name => s"${name}_vec"))

    val assembler = new VectorAssembler()
      .setInputCols(continueCols ++ (categoryCols map (name => s"${name}_vec")))
      .setOutputCol("features")

    val rfModel = new RandomForestClassifier()
      .setFeaturesCol("features")
      .setLabelCol("label")
      .setMaxDepth(10)
      .setNumTrees(100)
      .setMaxBins(100)

    println("*" * 30 + "模型参数" + "*" * 50)
    val paramSeq = rfModel.extractParamMap().toSeq
    for (item <- paramSeq) {
      println(item.param.name + ":" + item.value)
    }

    println("*" * 50 + "开始训练" + "*" * 50)
    val start = System.nanoTime()
    val pipeline = new Pipeline().setStages(Array(stringTransform(0), stringTransform(1), stringTransform(2), stringTransform(3)
      , stringTransform(4), stringTransform(5), stringTransform(6), stringTransform(7), stringTransform(8), stringTransform(9),
      stringTransform(10), stringTransform(11), stringTransform(12), stringTransform(13), stringTransform(14), stringTransform(15),
      oneHotTransform, assembler, rfModel)).fit(trainDF)
    println("*" * 50 + f"训练完成=>${(System.nanoTime() - start) / 1e9}%.2f second" + "*" * 50)

    pipeline.write.overwrite().save(saveHdfs)
    trainDF.unpersist()
  }

}
