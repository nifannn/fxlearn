package fxspark.embed

package object graph {
  case class NodeAttr(var neighbors: Array[(Long, Double)] = Array.empty[(Long, Double)],
                      var path: Array[Long] = Array.empty[Long],
                      var nodeName: String = "") extends Serializable

  case class EdgeAttr(var dstNeighbors: Array[Long] = Array.empty[Long],
                      var alias: Array[Int] = Array.empty[Int],
                      var accept: Array[Double] = Array.empty[Double]) extends Serializable

  lazy val createUndirectedEdge = (srcId: Long, dstId: Long, weight: Double) => {
    Array(
      (srcId, Array((dstId, weight))),
      (dstId, Array((srcId, weight)))
    )
  }

  lazy val createDirectedEdge = (srcId: Long, dstId: Long, weight: Double) => {
    Array(
      (srcId, Array((dstId, weight)))
    )
  }

}
