package fxspark.embed

import java.io.Serializable
import scala.collection.mutable.ArrayBuffer
import org.slf4j.{Logger, LoggerFactory}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SparkSession, Row}
import org.apache.spark.sql.functions.lit
import org.apache.spark.ml.feature.{Word2Vec, Word2VecModel}
import org.apache.spark.ml.linalg.DenseVector
import fxspark.sample.AliasSample


object Node2vecModel extends Serializable {
  val logger = LoggerFactory.getLogger(getClass.getName)

  case class Params(degree: Int = 30,
                    p: Double = 1.0,
                    q: Double = 1.0,
                    numWalks: Int = 10,
                    walkLength: Int = 80,
                    directed: Boolean = true,
                    weighted: Boolean = true,
                    embedDim: Int = 32,
                    maxIter: Int = 3,
                    minCount: Int = 5,
                    windowSize: Int = 5,
                    stepSize: Double = 0.025)

  var params: Params = null
  var edges: RDD[(String, String, Double)] = null
  var nodeProbs: RDD[(String, Array[String], Array[Double], Array[Int])] = null
  var edgeProbs: RDD[(String, Array[String], Array[Double], Array[Int])] = null
  var randomWalkPaths: RDD[ArrayBuffer[String]] = null
  var embeddings: DataFrame = null

  def setParams(degree: Int = 30, p: Double = 1.0, q: Double = 1.0, numWalks: Int = 10, walkLength: Int = 80,
                directed: Boolean = true, weighted: Boolean = true, embedDim: Int = 32, maxIter: Int = 3,
                minCount: Int = 5, windowSize: Int = 5, stepSize: Double = 0.025) : this.type = {
    params = Params(degree = degree, p = p, q = q, numWalks = numWalks, walkLength = walkLength,
      directed = directed, weighted = weighted, embedDim = embedDim, maxIter = maxIter, minCount = minCount,
      windowSize = windowSize, stepSize = stepSize)
    this
  }

  def buildGraph(spark: SparkSession, df: DataFrame) : this.type = {
    val maxDegree = spark.sparkContext.broadcast(params.degree)
    val weightedEdges = params.weighted match {
      case true => df.rdd.map(row => (row.getString(0), row.getString(1), row.getDouble(2)))
      case false => df.rdd.map(row => (row.getString(0), row.getString(1), 1.0))
    }
    val rawEdges = params.directed match {
      case true => weightedEdges.distinct()
      case false => weightedEdges.map(x => (x._2, x._1, x._3)).union(weightedEdges).distinct()
    }
    edges = rawEdges.map { case (srcNode, dstNode, weight) =>
      (srcNode, (dstNode, weight))
    }.groupByKey().flatMap { case (srcNode, dstIter) =>
      val dstArray = if (dstIter.size > maxDegree.value) {
        dstIter.toArray.sortWith{ case (left, right) => left._2 > right._2 }.slice(0, maxDegree.value)
      } else {
        dstIter.toArray
      }
      dstArray.map{ case (dstNode, weight) => (srcNode, dstNode, weight) }
    }.cache()
    edges.first()
    this
  }

  def getEdgeProbs(p: Double = 1.0, q: Double = 1.0)(dstNeighbors: Array[(String, Double)],
                                                     srcNode: String, srcNeighbors: Array[(String, Double)]): Array[Double] = {
    dstNeighbors.map { case (dstNode, weight) =>
      var unnormProb = weight / q
      if (srcNode == dstNode) unnormProb = weight / p
      else if (srcNeighbors.exists(_._1 == dstNode)) unnormProb = weight
      unnormProb
    }
  }

  def initTransitionProb(spark: SparkSession): this.type = {
    val p = spark.sparkContext.broadcast(params.p)
    val q = spark.sparkContext.broadcast(params.q)
    val nodeNeighbors = edges.map { case (srcNode, dstNode, weight) =>
      (srcNode, (dstNode, weight))
    }.groupByKey()
    nodeProbs = nodeNeighbors.map { case (srcNode, dstIter) =>
      val dstArray = dstIter.toArray
      val dstNodes = dstArray.map(_._1)
      val dstWeight = dstArray.map(_._2)
      val (accept, alias) = AliasSample.createAliasTable(dstWeight)
      (srcNode, dstNodes, accept, alias)
    }.cache()
    nodeProbs.first()
    edgeProbs = edges.map { case (srcNode, dstNode, weight) =>
      (srcNode, dstNode)
    }.join(nodeNeighbors).map { case (srcNode, (dstNode, srcNeighborsIter)) =>
      (dstNode, (srcNode, srcNeighborsIter))
    }.join(nodeNeighbors).map { case (dstNode, ((srcNode, srcNeighborsIter), dstNeighborsIter)) =>
      val srcNeighbors = srcNeighborsIter.toArray
      val dstNeighbors = dstNeighborsIter.toArray
      val probs = getEdgeProbs(p.value, q.value)(dstNeighbors, srcNode, srcNeighbors)
      val dstNodes = dstNeighbors.map(_._1)
      val (accept, alias) = AliasSample.createAliasTable(probs)
      (srcNode+"_"+dstNode, dstNodes, accept, alias)
    }.cache()
    edgeProbs.first()
    edges.unpersist(blocking = false)
    this
  }

  def randomWalkOnce(): RDD[ArrayBuffer[String]] = {
    val edgeProbsMap = edgeProbs.map { case (edge, dstNodes, accept, alias) =>
      (edge, (dstNodes, accept, alias))
    }
    var preWalk: RDD[ArrayBuffer[String]] = null
    var curWalk = nodeProbs.map { case (srcNode, dstNodes, accept, alias) =>
      val pathBuffer = new ArrayBuffer[String]()
      val nextNodeIdx = AliasSample.sample(accept, alias)
      pathBuffer.append(srcNode, dstNodes(nextNodeIdx))
      pathBuffer
    }.cache()
    curWalk.first()
    for (walkStep <- 0 until params.walkLength) {
      preWalk = curWalk
      curWalk = curWalk.map { case pathBuffer =>
        val preNode = pathBuffer(pathBuffer.length - 2)
        val curNode = pathBuffer.last
        (preNode+"_"+curNode, pathBuffer)
      }.leftOuterJoin(edgeProbsMap).map { case (edge, (pathBuffer, edgeProb)) =>
        if (!edgeProb.isEmpty) {
          val (dstNodes, accept, alias) = edgeProb.get
          val nextNodeIdx = AliasSample.sample(accept, alias)
          pathBuffer.append(dstNodes(nextNodeIdx))
        }
        pathBuffer
      }.cache()
      curWalk.first()
      preWalk.unpersist(blocking = false)
    }
    curWalk
  }

  def randomWalkAndSavePathAsTextFile(spark: SparkSession, path: String,
                                      overwrite: Boolean = true, pathDelimiter: String = " "): Unit = {
    import spark.implicits._
    for (iter <- 0 until params.numWalks) {
      randomWalkPaths = randomWalkOnce()
      val df = randomWalkPaths.map(_.mkString(pathDelimiter)).toDF()
      var writer = df.write
      if (iter == 0 && overwrite) {
        writer = writer.mode("overwrite")
      } else {
        writer = writer.mode("append")
      }
      writer.text(path)
      randomWalkPaths.unpersist(blocking = false)
    }
  }

  def randomWalkAndSavePathAsHiveTable(spark: SparkSession, tblName: String,
                                       overwrite: Boolean = true, pathDelimiter: String = " ",
                                       pathCol: String = "path", partitionCol: String = "", partitionVal: String = ""): Unit = {
    import spark.implicits._
    for (iter <- 0 until params.numWalks) {
      randomWalkPaths = randomWalkOnce()
      var df = randomWalkPaths.map(_.mkString(pathDelimiter)).toDF(pathCol)
      if (partitionCol.length > 0){
        df = df.withColumn(partitionCol, lit(partitionVal))
      }
      var writer = df.write
      if (iter == 0 && overwrite) {
        writer = writer.mode("overwrite")
      } else {
        writer = writer.mode("append")
      }
      if (partitionCol.length > 0){
        writer = writer.partitionBy(partitionCol)
      }
      writer.saveAsTable(tblName)
      randomWalkPaths.unpersist(blocking = false)
    }
  }

  def randomWalk(): this.type = {
    for (iter <- 0 until params.numWalks) {
      val randomWalk = randomWalkOnce()
      if (randomWalkPaths != null) {
        val prevRandomWalkPaths = randomWalkPaths
        randomWalkPaths = randomWalkPaths.union(randomWalk).cache
        randomWalkPaths.first
        prevRandomWalkPaths.unpersist(blocking = false)
      } else {
        randomWalkPaths = randomWalk
      }
    }
    this
  }

  def saveRandomPathAsTextFile(spark: SparkSession, path: String,
                               overwrite: Boolean = true, pathDelimiter: String = " "): this.type = {
    import spark.implicits._
    val df = randomWalkPaths.map(_.mkString(pathDelimiter)).toDF()
    val writer = overwrite match {
      case true => df.write.mode("overwrite")
      case false => df.write.mode("append")
    }
    writer.text(path)
    this
  }

  def saveRandomPathAsHiveTable(spark: SparkSession, tblName: String,
                                overwrite: Boolean = true, pathDelimiter: String = " ",
                                pathCol: String = "path", partitionCol: String = "", partitionVal: String = ""): this.type = {
    import spark.implicits._
    var df = randomWalkPaths.map(_.mkString(pathDelimiter)).toDF(pathCol)
    if (partitionCol.length > 0){
      df = df.withColumn(partitionCol, lit(partitionVal))
    }
    var writer = overwrite match {
      case true => df.write.mode("overwrite")
      case false => df.write.mode("append")
    }
    if (partitionCol.length > 0){
      writer = writer.partitionBy(partitionCol)
    }
    writer.saveAsTable(tblName)
    this
  }

  def trainEmbedding(saprk: SparkSession): this.type = {
    import saprk.implicits._
    val df = randomWalkPaths.map(_.toSeq).toDF()
    val word2vecModel = new Word2Vec().setMinCount(params.minCount).setWindowSize(params.windowSize)
        .setVectorSize(params.embedDim).setMaxIter(params.maxIter).setStepSize(params.stepSize).fit(df)
    embeddings = word2vecModel.getVectors
    this
  }

  def saveEmbeddingAsTextFile(spark: SparkSession, path: String,
                              overwrite: Boolean = true,
                              fieldDelimiter: String = "\t", embedDelimiter: String = " "): this.type = {
    import spark.implicits._
    val df = embeddings.rdd.map(row => row.getString(0)+fieldDelimiter+row.getAs[DenseVector](1).toArray.mkString(embedDelimiter)).toDF()
    val writer = overwrite match {
      case true => df.write.mode("overwrite")
      case false => df.write.mode("append")
    }
    writer.text(path)
    this
  }

  def saveEmbeddingAsHiveTable(spark: SparkSession, tblName: String,
                               overwrite: Boolean = true, embedDelimiter: String = " ",
                               nodeCol: String = "node", embedCol: String = "embedding",
                               partitionCol: String = "", partitionVal: String = ""): this.type = {
    import spark.implicits._
    var df = embeddings.rdd.map(row => (row.getString(0), row.getAs[DenseVector](1).toArray.mkString(embedDelimiter))).toDF(nodeCol, embedCol)
    if (partitionCol.length > 0){
      df = df.withColumn(partitionCol, lit(partitionVal))
    }
    var writer = overwrite match {
      case true => df.write.mode("overwrite")
      case false => df.write.mode("append")
    }
    if (partitionCol.length > 0){
      writer = writer.partitionBy(partitionCol)
    }
    writer.saveAsTable(tblName)
    this
  }
}