package fxspark.sample

import scala.collection.mutable.ArrayBuffer

object AliasSample {
  def createAliasTable(probs: Array[Double]) : (Array[Double], Array[Int]) = {
    val N = probs.length
    val accept = Array.fill(N)(0.0)
    val alias = Array.fill(N)(0)

    val small = new ArrayBuffer[Int]()
    val large = new ArrayBuffer[Int]()

    val probSum = probs.sum
    probs.zipWithIndex.foreach { case (prob, i) =>
      accept(i) = N * prob / probSum
      if (accept(i) < 1.0) {
        small.append(i)
      } else {
        large.append(i)
      }
    }
    while (small.nonEmpty && large.nonEmpty) {
      val smallIdx = small.remove(small.length - 1)
      val largeIdx = large.remove(large.length - 1)

      alias(smallIdx) = largeIdx
      accept(largeIdx) = accept(largeIdx) + accept(smallIdx) - 1.0

      if (accept(largeIdx) < 1.0) small.append(largeIdx)
      else large.append(largeIdx)
    }
    (accept, alias)
  }

  def sample(accept: Array[Double], alias: Array[Int]) : Int = {
    val N = accept.length
    val ix = math.floor(math.random * N).toInt
    if (math.random < accept(ix)) ix
    else alias(ix)
  }
}
