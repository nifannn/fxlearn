package spark_learn.util.client

import java.util.Properties
import org.apache.spark.sql.DataFrame
import redis.clients.jedis.Jedis

object RedisClient {
  def readKV(): Unit = {

  }

  def readList(): Unit = {

  }

  def writeKV(): Unit = {

  }

  def writeList(redisConf: Properties, df: DataFrame,
                keyCol: String, listCol: String, prefix: String ): Unit = {
    val host = redisConf.getProperty("host")
    val port = redisConf.getProperty("port").toInt
    val password = redisConf.getProperty("password")
    val timeout = redisConf.getProperty("timeout").toInt
    val expireTime = redisConf.getProperty("expire_time").toInt
    var tryTimes = redisConf.getProperty("try_times").toInt
    var flag = false

    while (tryTimes > 0 && !flag){
      try {
        df.repartition(500).foreachPartition(partition => {
          val rc = new Jedis(host, port, timeout)
          rc.auth(password)
          val pipe = rc.pipelined

          partition.foreach(row => {
            val key = prefix + row.getAs[Long](keyCol)
            val list = row.getAs[Seq[Long]](listCol)
            pipe.del(Array(key):_*)
            pipe.rpush(key, list.map(_.toString):_*)
            pipe.expire(key, expireTime)
            pipe.sync()
          })
        })
        flag = true
      } catch {
        case e: Exception =>
          flag = false
          tryTimes = tryTimes - 1
      }
    }
  }
}
