import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.types.{DoubleType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.ml.regression.LinearRegression

object LrRegression {

  def main(args: Array[String]): Unit = {
    /** Path of csv file */
    val dataPath = "/home/rcd/Dev/Scala_spark/data_csv/house.csv"

    val spark = SparkSession.builder
      .appName("House price regressor")
      .master("local")
      .getOrCreate()

    /** Print only ERROR log */
    spark.sparkContext.setLogLevel("ERROR")

    val schemaStruct = StructType(
      StructField("loyer", DoubleType) ::
        StructField("surface", DoubleType) :: Nil
    )

    /** Load data from csv file */
    val data = spark.read
      .option("header", true)
      .schema(schemaStruct)
      .csv(path = dataPath)

    data.show(5)

    /* Split data **/
    val Array(trainData, testData) = data.randomSplit(Array(0.8, 0.2))

    /** Dataframe doesn't have categorical variable we don't need StringIndexer */
    val labelColumn = "loyer"

    /** Create single vector with input column */
    val assembler = new VectorAssembler()
      .setInputCols(Array("surface"))
      .setOutputCol("features")

    /** Define linear regression estimator */
    val lr = new LinearRegression()
      .setMaxIter(100)
      .setRegParam(0.3)
      .setElasticNetParam(0.8)
      .setLabelCol("loyer")
      .setFeaturesCol("features")
      .setPredictionCol("Predicted" + labelColumn)

    val stages = Array(
      assembler,
      lr
    )

    val pipeline = new Pipeline().setStages(stages)

    val model = pipeline.fit(trainData)

    val preds = model.transform(testData)

    evaluateModel(labelColumn, preds)

    spark.stop()
  }


  def evaluateModel(labelCol: String, predictions: DataFrame): Unit = {

    /** Compute Root Mean Squared Error */
    val rmseEval = new RegressionEvaluator()
      .setLabelCol(labelCol)
      .setPredictionCol("Predicted" + labelCol)
      .setMetricName("rmse")
    val rmse = rmseEval.evaluate(predictions)

    /** Compute Mean Absolute Error */
    val maeEval = new RegressionEvaluator()
      .setLabelCol(labelCol)
      .setPredictionCol("Predicted" + labelCol)
      .setMetricName("mae")
    val mae = maeEval.evaluate(predictions)

    println("Model evaluation :")
    println("RMSE : " + rmse)
    println("MAE : " + mae)

  }
}
