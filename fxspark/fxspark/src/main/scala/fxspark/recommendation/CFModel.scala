package fxspark.recommendation

import org.apache.spark.sql.{SparkSession, DataFrame, Row}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions.lit

object CFModel {

  var similarityScore: DataFrame = null
  var topnResult: DataFrame = null

  def computeSimilarity(spark: SparkSession, df: DataFrame, modelType: String, similarFormula: String): this.type = {
    import spark.implicits._
    val matrixDF = modelType match {
      case "item" => df.rdd.map(row => (row.get(0).toString, row.getLong(1), row.getDouble(2))).toDF("user", "item", "rating")
      case "user" => df.rdd.map(row => (row.getLong(1), row.get(0).toString, row.getDouble(2))).toDF("item", "user", "rating")
    }
    similarityScore = similarFormula match {
      case "jaccard" => Similarity.jaccard(spark, matrixDF)
      case "jaccard-penalty" => Similarity.jaccard(spark, matrixDF, true)
      case "cosine" => Similarity.cosine(spark, matrixDF)
    }
    this
  }

  def computeTopn(spark: SparkSession, n: Int): this.type = {
    import spark.implicits._
    topnResult = similarityScore.rdd.map{ case row =>
      (row.getString(0), (row.getString(1), row.getDouble(2)))
    }.groupByKey().map { case (left, rightIter) =>
      val topN = rightIter.toArray.sortWith{ case (left, right) => left._2 > right._2 }.slice(0, n).map(_._1)
      (left, topN)
    }.toDF("id", "rec_list")
    this
  }

  def saveSimilarityAsTextFile(spark: SparkSession, path: String, overwrite: Boolean = false, fieldDelimiter: String = "\t"): this.type = {
    import spark.implicits._
    val df = similarityScore.rdd.map(row => String.join(fieldDelimiter,
      row.get(0).toString, row.get(1).toString, row.get(2).toString)).toDF()
    var writer = df.write
    writer = overwrite match {
      case true => writer.mode("overwrite")
      case false => writer.mode("append")
    }
    writer.text(path)
    this
  }

  def saveSimilarityAsHiveTable(tblName: String, overwrite: Boolean = false,
                                colI: String = "i", colJ: String = "j", scoreCol: String = "score",
                                partitionCol: String = "", partitionVal: String = ""): this.type = {
    var df = similarityScore.withColumnRenamed("i", colI).withColumnRenamed("j", colJ)
      .withColumnRenamed("score", scoreCol)
    if (partitionCol.length > 0) {
      df = df.withColumn(partitionCol, lit(partitionVal))
    }
    var writer = df.write
    writer = overwrite match {
      case true => writer.mode("overwrite")
      case false => writer.mode("append")
    }
    if (partitionCol.length > 0) {
      writer = writer.partitionBy(partitionCol)
    }
    writer.saveAsTable(tblName)
    this
  }

  def saveTopnAsTextFile(spark: SparkSession, path: String, overwrite: Boolean = false,
                         fieldDelimiter: String = "\t", collectionDelimiter: String = ","): this.type = {
    import spark.implicits._
    val df = topnResult.rdd.map(row => String.join(fieldDelimiter, row.get(0).toString,
      String.join(collectionDelimiter, row.getList(1).toArray.map(_.toString):_*))).toDF()
    var writer = df.write
    writer = overwrite match {
      case true => writer.mode("overwrite")
      case false => writer.mode("append")
    }
    writer.text(path)
    this
  }

  def saveTopnAsHiveTable(tblName: String, overwrite: Boolean = false,
                          mainCol: String = "id", recCol: String = "rec_list",
                          partitionCol: String = "", partitionVal: String = ""): this.type = {
    var df = topnResult.withColumnRenamed("id", mainCol).withColumnRenamed("rec_list", recCol)
    if (partitionCol.length > 0) {
      df = df.withColumn(partitionCol, lit(partitionVal))
    }
    var writer = df.write
    writer = overwrite match {
      case true => writer.mode("overwrite")
      case false => writer.mode("append")
    }
    if (partitionCol.length > 0) {
      writer = writer.partitionBy(partitionCol)
    }
    writer.saveAsTable(tblName)
    this
  }
}
