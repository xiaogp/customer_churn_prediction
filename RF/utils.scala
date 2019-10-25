import java.io.FileInputStream
import java.time.format.DateTimeFormatter
import java.time.{LocalDate, ZoneId}
import java.util.Properties

import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{col, udf}

object utils {

  /**
    * 获取配置文件
    *
    * @param proPath
    * @return
    */
  def getProPerties(proPath: String) = {
    val properties: Properties = new Properties()
    properties.load(new FileInputStream(proPath))
    properties
  }


  /**
    * 求前多少天的日期
    *
    * @param dayDiff   时间窗口
    * @param startDate 起始时间
    * @return
    */
  def getDiffDatetime(dayDiff: Int, startDate: Any = None): String = {
    var date = LocalDate.now(ZoneId.systemDefault())
    if (startDate != None) {
      date = LocalDate.parse(startDate.toString, DateTimeFormatter.ofPattern("yyyy-MM-dd"))
    }
    date.plusDays(-dayDiff).toString()
  }

  /**
    * 计算recall和precison，阈值为0.5
    *
    * @param data
    * @param labelCol
    * @param predCol
    */
  def PREvaluation(data: DataFrame, labelCol: String, predCol: String): Unit = {
    val dataPred = data.select(col(predCol).cast("Double"), col(labelCol).cast("Double"))
      .rdd.map(row => (row.getDouble(0), row.getDouble(1)))
    val prMetrics = new MulticlassMetrics(dataPred)
    println(f"accuracy: ${prMetrics.accuracy}%.3f")
    println(f"precision: ${prMetrics.precision(1)}%.3f")
    println(f"recall: ${prMetrics.recall(1)}%.3f")
    println(f"fMeasure: ${prMetrics.fMeasure(1)}%.3f")
  }

  /**
    * 计算AUC
    *
    * @param data     预测数据
    * @param labelCol 实际标签
    * @param predCol  预测标签
    * @return
    */
  def AUCEvaluation(data: DataFrame, labelCol: String, predCol: String): Unit = {
    val metrics = new BinaryClassificationEvaluator()
      .setLabelCol(labelCol)
      .setRawPredictionCol(predCol)
      .setMetricName("areaUnderROC")
    val auc = metrics.evaluate(data)
    println(f"AreaUnderROC: ${auc}%.3f")
  }

  /**
    * 获得正类预测概率
    *
    * @param data   原始数据
    * @param preCol 预测列
    * @return
    */
  def getProba(data: DataFrame, preCol: String): DataFrame = {
    val probaFunc = udf((proba: Vector) => (proba(1)))
    data.withColumn("predProba", probaFunc(col(preCol)))
  }

}