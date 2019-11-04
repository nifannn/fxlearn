package spark_learn.util.data_pipeline

import java.io.InputStream
import java.util.Properties
import org.apache.spark.sql.{DataFrame, SparkSession}
import spark_learn.util.client.RedisClient

object Hive2Redis {
  val prop = new Properties()
  val inputStream = this.getClass.getClassLoader.getResourceAsStream("config/hive2redis.properties")
  prop.load(inputStream)

  private val redisHost = prop.getProperty("redis.host")
  private val redisPort = prop.getProperty("redis.port")
  private val redisPassword = prop.getProperty("redis.password")
  private val redisTimeout = prop.getProperty("redis.timeout")
  private val redisExpireTime = prop.getProperty("redis.expire_time")
  private val redisTryTimes = prop.getProperty("redis.try_times")
  private val dfKeyCol = prop.getProperty("key_col")
  private val dfListCol = prop.getProperty("list_col")
  private val redisPrefix = prop.getProperty("prefix")

  val hqlStream = this.getClass.getClassLoader.getResourceAsStream("hql/hive2redis.hql")
  private val hql = scala.io.Source.fromInputStream(hqlStream).mkString

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.
      builder()
      .appName(s"${this.getClass.getSimpleName}")
      .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .config("spark.sql.broadcastTimeout", "36000")
      .enableHiveSupport()
      .getOrCreate()

    val df = spark.sql(hql)

    val redisConf = new Properties()
    redisConf.setProperty("host", redisHost)
    redisConf.setProperty("port", redisPort)
    redisConf.setProperty("password", redisPassword)
    redisConf.setProperty("timeout", redisTimeout)
    redisConf.setProperty("expire_time", redisExpireTime)
    redisConf.setProperty("try_times", redisTryTimes)

    RedisClient.writeList(redisConf, df, dfKeyCol, dfListCol, redisPrefix)
  }
}