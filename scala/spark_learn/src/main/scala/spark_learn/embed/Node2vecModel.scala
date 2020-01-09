package spark_learn.embed

import java.io.Serializable
import scala.util.Try
import scala.collection.mutable.ArrayBuffer
import org.slf4j.{Logger, LoggerFactory}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SparkSession, Row, SaveMode}
import org.apache.spark.graphx.{EdgeTriplet, Graph, VertexId, Edge}
import org.apache.spark.SparkContext
import spark_learn.embed.graph.{NodeAttr, EdgeAttr, createDirectedEdge, createUndirectedEdge}
import spark_learn.sample.AliasSample


object Node2vecModel extends Serializable {
  val logger = LoggerFactory.getLogger(getClass.getName)

  case class Params(degree: Int = 30,
                    p: Double = 1.0,
                    q: Double = 1.0,
                    numWalks: Int = 10,
                    walkLength: Int = 80,
                    numPartition: Int = 200)

  var sc: SparkContext = null
  var spark: SparkSession = null
  var params: Params = null
  var node2id: RDD[(String, Long)] = null
  var indexedEdges: RDD[Edge[EdgeAttr]] = _
  var indexedNodes: RDD[(VertexId, NodeAttr)] = _
  var graph: Graph[NodeAttr, EdgeAttr] = _
  var randomWalkPaths: RDD[(Long, ArrayBuffer[Long])] = null

  def setup(spark: SparkSession, params: Params) : this.type = {
    this.spark = spark
    this.sc = spark.sparkContext
    this.params = params
    this
  }

  def buildGraph(df: DataFrame) : this.type = {
    val maxDegree = sc.broadcast(params.degree)
    val rawTriplets = df.rdd.map{ case Row(srcNode: String, dstNode: String, weight: Double) => (srcNode, dstNode, weight)}
    val indexedTriplets = indexingTriplets(rawTriplets)
    val nodeNeighbors = indexedTriplets.flatMap { case (srcId, dstId, weight) =>
      createDirectedEdge.apply(srcId, dstId, weight)
    }.reduceByKey(_++_).map{ case (nodeId, neighbors: Array[(Long,Double)]) =>
      var neighbors_ = neighbors
      if (neighbors_.length > maxDegree.value) {
        neighbors_ = neighbors.sortWith{ case (left, right) => left._2 > right._2 }.slice(0, maxDegree.value)
      }
      (nodeId, neighbors_)}
    indexedNodes = getId2Node.leftOuterJoin(nodeNeighbors).map {
      case (nodeId, (nodeName: String, neighbors: Option[Array[(Long, Double)]])) =>
        (nodeId, NodeAttr(neighbors = neighbors.getOrElse(Array.empty[(Long, Double)]), nodeName = nodeName))
    }.repartition(params.numPartition).cache
    indexedEdges = nodeNeighbors.flatMap { case (srcId, linkedNodes) =>
      linkedNodes.map { case (dstId, weight) =>
        Edge(srcId, dstId, EdgeAttr())
      }
    }.repartition(params.numPartition).cache
    graph = Graph(indexedNodes, indexedEdges)
    this
  }

  def indexingTriplets(rawTriplets: RDD[(String, String, Double)]): RDD[(Long, Long, Double)] = {
    this.node2id = createNode2Id(rawTriplets)

    rawTriplets.map { case (src, dst, weight) =>
      (src, (dst, weight))
    }.join(node2id).map { case (src, (edge: (String, Double), srcIndex: Long)) =>
      try {
        val (dst: String, weight: Double) = edge
        (dst, (srcIndex, weight))
      } catch {
        case e: Exception => null
      }
    }.filter(_!=null).join(node2id).map { case (dst, (edge: (Long, Double), dstIndex: Long)) =>
      try {
        val (srcIndex, weight) = edge
        (srcIndex, dstIndex, weight)
      } catch {
        case e: Exception => null
      }
    }.filter(_!=null)
  }

  def initTransitionProb(): this.type = {
    val P = sc.broadcast(params.p)
    val Q = sc.broadcast(params.q)

    graph = graph.mapVertices[NodeAttr] { case (vertexId, nodeAttrs) =>
        if (nodeAttrs.neighbors.length > 0){
          val (accept, alias) = AliasSample.createAliasTable(nodeAttrs.neighbors.map(_._2))
          val nextNodeIdx = AliasSample.sample(accept, alias)
          nodeAttrs.path = Array(vertexId, nodeAttrs.neighbors(nextNodeIdx)._1)
        }
        nodeAttrs
    }.mapTriplets { edgeTriplet: EdgeTriplet[NodeAttr, EdgeAttr] =>
      if (edgeTriplet.dstAttr.neighbors.length > 0){
        val edgeProbs = getEdgeProbs(P.value, Q.value)(edgeTriplet)
        val (accept, alias) = AliasSample.createAliasTable(edgeProbs)
        edgeTriplet.attr.accept = accept
        edgeTriplet.attr.alias = alias
        edgeTriplet.attr.dstNeighbors = edgeTriplet.dstAttr.neighbors.map(_._1)
      }
      edgeTriplet.attr
    }.cache
    this
  }

  def randomWalk(): this.type = {
    val edge2attr = graph.triplets.map { edgeTriplet =>
      (s"${edgeTriplet.srcId}_${edgeTriplet.dstId}", edgeTriplet.attr)
    }.repartition(params.numPartition).cache
    edge2attr.first

    for (iter <- 0 until params.numWalks) {
      var prevWalk: RDD[(Long, ArrayBuffer[Long])] = null
      var randomWalk = graph.vertices.filter(_._2.neighbors.length>0).map { case (nodeId, nodeAttr) =>
          val pathBuffer = new ArrayBuffer[Long]()
          pathBuffer.append(nodeAttr.path:_*)
        (nodeId, pathBuffer)
      }.cache
      randomWalk.first
      graph.unpersist(blocking = false)
      graph.edges.unpersist(blocking = false)

      for (walkCount <- 0 until params.walkLength) {
        prevWalk = randomWalk
        randomWalk = randomWalk.map { case (srcNodeId, pathBuffer) =>
            val prevNodeId = pathBuffer(pathBuffer.length - 2)
            val currentNodeId = pathBuffer.last
          (s"${prevNodeId}_${currentNodeId}", (srcNodeId, pathBuffer))
        }.join(edge2attr).map { case (edge, ((srcNodeId, pathBufffer), attr)) =>
            if (attr.dstNeighbors.length>0) {
              val nextNodeIndex = AliasSample.sample(attr.accept, attr.alias)
              val nextNodeId = attr.dstNeighbors(nextNodeIndex)
              pathBufffer.append(nextNodeId)
            }
          (srcNodeId, pathBufffer)
        }.cache
        randomWalk.first
        prevWalk.unpersist(blocking = false)
      }

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

  def createNode2Id(triplets: RDD[(String, String, Double)]) : RDD[(String, Long)] = triplets.flatMap{ case (src, dst, weight) =>
  Try(Array(src, dst)).getOrElse(Array.empty[String])
}.distinct().zipWithIndex()

  def getEdgeProbs(p: Double = 1.0, q: Double = 1.0)(edgeTriplet: EdgeTriplet[NodeAttr, EdgeAttr]): Array[Double] = {
    val srcId = edgeTriplet.srcId
    val srcNeighbors = edgeTriplet.srcAttr.neighbors
    val dstNeighbors = edgeTriplet.dstAttr.neighbors
    dstNeighbors.map { case (dstNodeId, weight) =>
        var unnormProb = weight / q
        if (srcId == dstNodeId) unnormProb = weight / p
        else if (srcNeighbors.exists(_._1 == dstNodeId)) unnormProb = weight
        unnormProb
    }
  }

  def getId2Node = this.node2id.map{ case (node, index) => (index, node) }

  def getRandomPaths = {
    randomWalkPaths.map { case (srcNodeId, pathBuffer) =>
      pathBuffer.toArray }.zipWithIndex.flatMap { case (walkPath, pathId) =>
      walkPath.zipWithIndex.map { case (nodeId, posId) =>
        (nodeId, (posId, pathId))
      }
    }.join(getId2Node).map { case (nodeId, ((posId, pathId), nodeName)) =>
      (pathId, Array((nodeName, posId)))
    }.reduceByKey(_++_).map(_._2.sortBy(_._2).map(_._1))
  }

  def saveRandomPathAsTextFile(path: String, sep: String = " "): this.type = {
    getRandomPaths.map(_.mkString(sep)).saveAsTextFile(path)
    this
  }

  def saveRandomPathAsHiveTable(tblName: String, column: String = "sequence", sep: String = " "): this.type = {
    val dupSpark = spark
    import dupSpark.implicits._
    getRandomPaths.map(_.mkString(sep)).toDF(column).write.mode(SaveMode.Overwrite).saveAsTable(tblName)
    this
  }
}