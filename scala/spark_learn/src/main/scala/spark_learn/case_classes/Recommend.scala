package spark_learn.case_classes

case class Rating(userId: String, itemId: Int, rating: Float)

case class ItemCtr(itemId: Int, ctr: Float)