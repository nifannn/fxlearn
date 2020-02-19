package fxspark.recommendation

import scala.collection.mutable.ArrayBuffer
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions.lit
import util.Random.shuffle

object SwingModel {
  var edges: RDD[(String, Long)] = null
  var itemSimilarity: RDD[(Long, Long, Double)] = null
  var topnResult: RDD[(Long, Array[Long])] = null

  def buildGraph(df: DataFrame, maxUser: Int): this.type = {
    val userListByItem = df.rdd.map(row => (row.getLong(1), row.get(0).toString)).groupByKey().map(x => (x._1, x._2.toArray))
    edges = userListByItem.map { case (item, userArray) =>
      if (userArray.length > maxUser) {
        (item, shuffle(userArray.toList).take(maxUser).toArray)
      }else {
        (item, userArray)
      }
    }.flatMap { case (item, userArray) =>
      userArray.map(user => (user, item))
    }.cache()
    edges.first()
    this
  }

  def computeItemSimilarity(): this.type = {
    val userPairByItemPair = edges.groupByKey().filter(_._2.size >= 2).flatMap { case (user, itemIter) =>
      val itemArray = itemIter.toArray.sorted
      val itemPairByUser = new ArrayBuffer[(String, String)]()
      for (i <- 0 until itemArray.length-1) {
        for (j <- i+1 until itemArray.length) {
          itemPairByUser.append((itemArray(i)+"_"+itemArray(j), user))
        }
      }
      itemPairByUser
    }.groupByKey().filter(_._2.size >= 2).flatMap { case (itemPair, userIter) =>
      val userArray = userIter.toArray.sorted
      val result = new ArrayBuffer[(String, String)]()
      for (i <- 0 until userArray.length-1) {
        for (j <- i+1 until userArray.length) {
          result.append((itemPair, userArray(i)+","+userArray(j)))
        }
      }
      result
    }
    val itemCountByUserPair = edges.map(x => (x._2, x._1)).groupByKey().filter(_._2.size >=2).flatMap { case (item, userIter) =>
      val userArray = userIter.toArray.sorted
      val userPair = new ArrayBuffer[(String, Long)]()
      for (i <- 0 until userArray.length-1) {
        for (j <- i+1 until userArray.length) {
          userPair.append((userArray(i)+","+userArray(j), 1))
        }
      }
      userPair
    }.reduceByKey(_+_)
    val itemPairSimilarity = userPairByItemPair.map {case (itemPair, userPair) =>
      (userPair, itemPair)
    }.join(itemCountByUserPair).map{ case (userPair, (itemPair, itemCount)) =>
      (itemPair, 1.0/(1+itemCount))
    }.reduceByKey(_+_).map { case (itemPair, score) =>
      val splitedItemPair = itemPair.split("_")
      val itemI = splitedItemPair(0).toLong
      val itemJ = splitedItemPair(1).toLong
      (itemI, itemJ, score)
    }.cache()
    itemPairSimilarity.first()
    edges.unpersist(blocking = false)
    itemSimilarity = itemPairSimilarity.union(itemPairSimilarity.map(x => (x._2, x._1, x._3))).cache()
    itemSimilarity.first()
    this
  }

  def computeTopn(n: Int): this.type = {
    topnResult = itemSimilarity.map(x => (x._1, (x._2, x._3))).groupByKey().map(x => (x._1, x._2.toArray)).map {case (item, itemScore) =>
      val topnItem = itemScore.sortWith{ case (left, right) => left._2 > right._2 }.slice(0, n).map(_._1)
      (item, topnItem)
    }.cache()
    topnResult.first()
    itemSimilarity.unpersist(blocking = false)
    this
  }

  def saveTopnAsTextFile(spark:SparkSession, path: String, overwrite: Boolean = false,
                         fieldDelimiter: String = "\t", collectionDelimiter: String = ","): this.type = {
    import spark.implicits._
    val df = topnResult.map(x => String.join(fieldDelimiter, x._1.toString, String.join(collectionDelimiter, x._2.map(_.toString):_*))).toDF()
    var writer = df.write
    writer = overwrite match {
      case true => writer.mode("overwrite")
      case false => writer.mode("append")
    }
    writer.text(path)
    this
  }

  def saveTopnAsHiveTable(spark: SparkSession, tblName: String,
                          itemCol: String = "item", itemListCol: String = "item_list",
                          overwrite: Boolean = false,
                          partitionCol: String = "", partitionVal: String = ""): this.type = {
    import spark.implicits._
    var df = topnResult.toDF(itemCol, itemListCol)
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

  def saveItemSimilarityAsTextFile(spark: SparkSession, path: String,
                                   overwrite: Boolean = false, fieldDelimiter: String = "\t"): this.type = {
    import spark.implicits._
    val df = itemSimilarity.map(x => String.join(fieldDelimiter, x._1.toString, x._2.toString, x._3.toString)).toDF()
    var writer = df.write
    writer = overwrite match {
      case true => writer.mode("overwrite")
      case false => writer.mode("append")
    }
    writer.text(path)
    this
  }

  def saveItemSimilarityAsHiveTable(spark: SparkSession, tblName: String,
                                    itemICol: String = "item_i", itemJCol: String = "item_j", scoreCol: String = "score",
                                    overwrite: Boolean = false,
                                    partitionCol: String = "", partitionVal: String = ""): this.type = {
    import spark.implicits._
    var df = itemSimilarity.toDF(itemICol, itemJCol, scoreCol)
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