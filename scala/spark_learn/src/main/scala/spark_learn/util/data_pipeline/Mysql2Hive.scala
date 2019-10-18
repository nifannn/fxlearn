package spark_learn.util.data_pipeline

import java.io.InputStream
import java.util.Properties
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import org.apache.spark.{SparkConf, SparkContext}

object Mysql2Hive {
  val prop = new Properties()
  val inputStream = this.getClass.getClassLoader.getResourceAsStream("mysql2hive.properties")
  prop.load(inputStream)

  private val mysqlUrl = prop.getProperty("jdbc.url")
  private val mysqlUser = prop.getProperty("jdbc.user")
  private val mysqlPassword = prop.getProperty("jdbc.password")
  private val mysqlTable = prop.getProperty("jdbc.table")
  private val mysqlDriver = prop.getProperty("jdbc.driver")
  private val hiveTable = prop.getProperty("hive.table")

  def main(args: Array[String]): Unit = {

  }
}