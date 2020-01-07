package spark_learn

package object recommendation {
  case class Rating(userId: String, itemId: Int, rating: Float)

  case class ItemCtr(itemId: Int, ctr: Float)
}
