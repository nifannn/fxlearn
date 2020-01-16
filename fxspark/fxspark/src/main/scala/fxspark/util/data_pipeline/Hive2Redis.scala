package fxspark.util.data_pipeline

import java.io.InputStream
import java.util.Properties
import scala.io.Source
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.slf4j.{Logger, LoggerFactory}
import scopt.OParser
import fxspark.util.client.RedisClient
import fxspark.Banner

object Hive2Redis {
  val prop = new Properties()
  val inputStream = this.getClass.getClassLoader.getResourceAsStream("redis_config.properties")
  prop.load(inputStream)

  private val redisHost = prop.getProperty("redis.host")
  private val redisPort = prop.getProperty("redis.port")
  private val redisPassword = prop.getProperty("redis.password")
  private val redisTimeout = prop.getProperty("redis.timeout")
  private val redisExpireTime = prop.getProperty("redis.expire_time")
  private val redisTryTimes = prop.getProperty("redis.try_times")

  val logger = LoggerFactory.getLogger(getClass.getName)

  case class Config(
                   hqlFile: String = "",
                   redisPrefix: String = ""
                   )

  val builder = OParser.builder[Config]
  val parser = {
    import builder._
    OParser.sequence(
      programName("spark-submit --class spark_learn.util.data_pipeline.Hive2Redis "),
      head("spark_learn", "1.0"),
      opt[String]('f', "hql")
        .required()
        .valueName("<hql_file>")
        .action((x, c) => c.copy(hqlFile = x))
        .text("hql file"),
      opt[String]('p', "prefix")
        .required()
        .action((x, c) => c.copy(redisPrefix = x))
        .text("redis prefix")
    )
  }

  def main(args: Array[String]): Unit = {
    logger.info(Banner.sparkLearn)
    val config = OParser.parse(parser, args, Config()) match {
      case Some(config) => {
        logger.info("hql file : "+config.hqlFile)
        logger.info("prefix : "+config.redisPrefix)
        config
      }
      case _ => {
        logger.warn("inappropriate parameters, exit")
        sys.exit(1)
      }
    }

    val bufferedSource = Source.fromFile(config.hqlFile)
    val hql = bufferedSource.getLines.mkString
    bufferedSource.close

    logger.info("hql : "+hql)

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

    logger.info("redis host : "+redisHost)

    RedisClient.writeList(redisConf, df, config.redisPrefix)
  }
}