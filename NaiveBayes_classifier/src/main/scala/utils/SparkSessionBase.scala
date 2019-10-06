package utils
import org.apache.spark.sql.SparkSession

trait SparkSessionBase {

  val sparkSession = SparkSession.builder
    .master("local[*]")
    .appName("Naive Bayes classifier")
    .getOrCreate()
}
