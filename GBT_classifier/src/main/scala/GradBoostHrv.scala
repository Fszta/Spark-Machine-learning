import utils.SparkSessionBase
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.sql.types.{DoubleType, StringType, StructField, StructType}
import org.apache.spark.ml.classification.GBTClassifier


object GradBoostHrv extends SparkSessionBase {
  /** Classify real heart rate variability data using
   *  Gradient boosting tree. This is a binary classification
   *  problem, the goal is to predict if cardiac data correspond to
   *  a stress or rest state
   * */
  def main(args: Array[String]): Unit = {

    val pathToData = "/home/rcd/Dev/Scala_spark/data_csv/hrv.csv"
    sparkSession.sparkContext.setLogLevel("ERROR")

    val schemaStruct = StructType(
      StructField("bpm", DoubleType) ::
      StructField("rmssd", DoubleType) ::
      StructField("bsv", DoubleType) ::
      StructField("sdnn", DoubleType) ::
      StructField("state", StringType) :: Nil
    )
    /** Load data & apply our schema structure */
    val dataHrv = sparkSession.read
      .option("header", true)
      .schema(schemaStruct)
      .csv(pathToData)

    dataHrv.show(5)

    /** Split data */
    val Array(trainData, testData) = dataHrv.randomSplit(Array(0.8,0.2))

    /** Index label */
    val classIndexer = new StringIndexer()
      .setInputCol("state")
      .setOutputCol("label")
      .fit(dataHrv)

    /** Create single vector containing features */
    val assembler = new VectorAssembler()
      .setInputCols(Array("bpm","rmssd","bsv","sdnn"))
      .setOutputCol("features")


    val gbtClassifier = new GBTClassifier()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setPredictionCol("Predicted state")
      .setMaxBins(10)
      .setMaxIter(10)

    val stages = Array(
      assembler,
      classIndexer,
      gbtClassifier
    )

    val pipeline = new Pipeline().setStages(stages)

    val model = pipeline.fit(trainData)

    val predictions = model.transform(testData)

    /** Evaluate model using area under ROC*/
    val evaluator = new BinaryClassificationEvaluator()
      .setLabelCol("label")
      .setRawPredictionCol("Predicted state")
      .setMetricName("areaUnderROC")

    val auc = evaluator.evaluate(predictions)

    println("AUC :" + auc)

  }
}
