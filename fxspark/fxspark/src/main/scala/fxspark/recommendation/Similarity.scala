package fxspark.recommendation

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.rdd.RDD
import scala.collection.mutable.ArrayBuffer

object Similarity {
  def cosine(spark:SparkSession, df: DataFrame): DataFrame = {
    import spark.implicits._
    val ratingRDD = df.rdd.map(row => (row.get(0).toString, row.get(1).toString, row.getDouble(2)))
    val rightPairDot = ratingRDD.map(x => (x._1, (x._2, x._3))).groupByKey().flatMap { case (left, rightRatingIter) =>
      val rightRatingArray = rightRatingIter.toArray.sortBy(_._1)
      val productByRightPair = new ArrayBuffer[(String, Double)]()
      for (i <- 0 until rightRatingArray.length-1) {
        for (j <- i+1 until rightRatingArray.length) {
          productByRightPair.append((rightRatingArray(i)._1+"_"+rightRatingArray(j)._1, rightRatingArray(i)._2 * rightRatingArray(j)._2))
        }
      }
      productByRightPair
    }.reduceByKey(_+_)
    val rightNorm = ratingRDD.map(x => (x._2, x._3 * x._3)).reduceByKey(_ + _).map(x => (x._1, math.sqrt(x._2)))
    val similarityScore = rightPairDot.map { case (rightPair, product) =>
      val pair = rightPair.split("_")
      val rightI = pair(0)
      val rightJ = pair(1)
      (rightI, (rightJ, product))
    }.join(rightNorm).map { case (rightI, ((rightJ, product), normI)) =>
      (rightJ, (rightI, normI, product))
    }.join(rightNorm).map { case (rightJ, ((rightI, normI, product), normJ)) =>
      (rightI, rightJ, product / (normI * normJ))
    }
    similarityScore.union(similarityScore.map(x => (x._2, x._1, x._3))).toDF("i", "j", "score")
  }

  def jaccard(spark: SparkSession, df: DataFrame, penalty: Boolean = false): DataFrame = {
    import spark.implicits._
    val pairRDD = df.rdd.map(row => (row.get(0).toString, row.get(1).toString))
    val rightPariCount = pairRDD.groupByKey().flatMap { case (left, rightIter) =>
      val rightArray = rightIter.toArray.sorted
      val rightPair = new ArrayBuffer[(String, Double)]()
      for (i <- 0 until rightArray.length-1) {
        for (j <- i+1 until rightArray.length){
          if (penalty) rightPair.append((rightArray(i)+"_"+rightArray(j), 1.0 / math.log(1.0 + rightArray.length)))
          else rightPair.append((rightArray(i)+"_"+rightArray(j), 1.0))
        }
      }
      rightPair
    }.reduceByKey(_+_)
    val leftArrayByRight = pairRDD.map(x => (x._2, x._1)).groupByKey()
    val similarityScore = rightPariCount.map { case (rightPair, cnt) =>
      val pair = rightPair.split("_")
      val rightI = pair(0)
      val rightJ = pair(1)
      (rightI, (rightJ, cnt))
    }.join(leftArrayByRight).map { case (rightI, ((rightJ, cnt), leftArrayOfRightI)) =>
      (rightJ, (rightI, cnt, leftArrayOfRightI))
    }.join(leftArrayByRight).map { case (rightJ, ((rightI, cnt, leftIterOfRightI), leftIterOfRightJ)) =>
      val leftArrayOfRightI = leftIterOfRightI.toArray
      val leftArrayOfRightJ = leftIterOfRightJ.toArray
      val unionCnt = (leftArrayOfRightI ++ leftArrayOfRightJ).distinct.length
      val score = cnt * 1.0 / unionCnt
      (rightI, rightJ, score)
    }
    similarityScore.union(similarityScore.map(x => (x._2, x._1, x._3))).toDF("i", "j", "score")
  }
}


