import utils.SparkSessionBase
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.sql.types.{DoubleType, StringType, StructField, StructType, IntegerType}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator


object NbClassifier extends SparkSessionBase {
  /** Classify real heart rate variability data using
   *  Gradient boosting tree. This is a binary classification
   *  problem, the goal is to predict if cardiac data correspond to
   *  a stress or rest state
   * */
  def main(args: Array[String]): Unit = {

    val dataPath = "/home/rcd/Dev/Scala_spark/data_csv/Iris.csv"

    val spark = SparkSession.builder
      .appName("Iris classification Naive Bayes")
      .master("local")
      .getOrCreate()

    /** Print only ERROR log */
    spark.sparkContext.setLogLevel("ERROR")

    val schemaStruct = StructType(
      StructField("Id", IntegerType) ::
        StructField("SepalLengthCm", DoubleType) ::
        StructField("SepalWidthCm", DoubleType) ::
        StructField("PetalLengthCm", DoubleType) ::
        StructField("PetalWidthCm", DoubleType) ::
        StructField("Species", StringType) :: Nil
    )

    val df = spark.read
      .option("header", true)
      .schema(schemaStruct)
      .csv(path=dataPath)

    val new_df = df.drop("Id")

    val Array(trainData,testData) = new_df.randomSplit(Array(0.8,0.2))

    val labelColName = "Label"

    val labelIndexer = new StringIndexer()
      .setInputCol("Species")
      .setOutputCol(labelColName)

    val assembler = new VectorAssembler()
      .setInputCols(Array("SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"))
      .setOutputCol("features")

    val NbClassifier = new NaiveBayes()
      .setLabelCol(labelColName)
      .setFeaturesCol("features")
      .setPredictionCol("Predicted " + labelColName)

    val stages = Array(
      assembler,
      labelIndexer,
      NbClassifier
    )

    val pipeline = new Pipeline().setStages(stages)

    val model = pipeline.fit(trainData)

    val predictions = model.transform(testData)
    
    /** Evaluate model using area under ROC*/
    val accuracyEvaluator = new MulticlassClassificationEvaluator()
      .setLabelCol(labelColName)
      .setPredictionCol("Predicted " + labelColName)
      .setMetricName("accuracy")
    val accuracy = accuracyEvaluator.evaluate(predictions)

    val f1Evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol(labelColName)
      .setPredictionCol("Predicted " + labelColName)
      .setMetricName("f1")
    val f1Score = f1Evaluator.evaluate(predictions)

    val recallEvaluator = new MulticlassClassificationEvaluator()
      .setLabelCol(labelColName)
      .setPredictionCol("Predicted " + labelColName)
      .setMetricName("weightedRecall")
    val recall = recallEvaluator.evaluate(predictions)

    val precisionEvaluator = new MulticlassClassificationEvaluator()
      .setLabelCol(labelColName)
      .setPredictionCol("Predicted " + labelColName)
      .setMetricName("weightedPrecision")
    val precision =precisionEvaluator.evaluate(predictions)

    println("Accuracy : " + accuracy)
    println("Recall : " + recall)
    println("Precision : " + precision)
    println("f1 score : " + f1Score)

  }
}
