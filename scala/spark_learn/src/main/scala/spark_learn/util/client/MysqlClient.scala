package spark_learn.util.client

import java.util.Properties

import org.apache.spark.sql.{DataFrame, SparkSession}

object MysqlClient {
  def read(spark: SparkSession, prop: Properties): DataFrame = {
    spark.read.format("jdbc")
      .option("url", prop.getProperty("jdbc.url"))
      .option("driver", prop.getProperty("jdbc.driver"))
      .option("user", prop.getProperty("jdbc.user"))
      .option("password", prop.getProperty("jdbc.password"))
      .option("dbtable", prop.getProperty("jdbc.table")).load()
  }

  def write(prop: Properties, df: DataFrame): Unit = {

  }
}
