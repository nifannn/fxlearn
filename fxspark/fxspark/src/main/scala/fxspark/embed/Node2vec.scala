package fxspark.embed

import org.slf4j.{Logger, LoggerFactory}
import scala.io.Source
import org.apache.spark.sql.{DataFrame, SparkSession}
import scopt.OParser
import fxspark.Banner

object Node2vec {
  val logger = LoggerFactory.getLogger(getClass.getName)

  case class Config(
                   hqlFile: String = "",
                   writeTable: String = "",
                   column: String = "sequence",
                   degree: Int = 30,
                   p: Double = 1.0,
                   q: Double = 1.0,
                   numWalks: Int = 10,
                   walkLength: Int = 80,
                   numPartition: Int = 200
                   )

  val builder = OParser.builder[Config]
  val parser = {
    import builder._
    OParser.sequence(
      programName("spark-submit --class spark_learn.embed.Node2vec "),
      head("spark_learn", "1.0"),
      opt[String]('f', "hql")
        .required()
        .valueName("<hql_file>")
        .action((x, c) => c.copy(hqlFile = x))
        .text("hql file"),
      opt[String]('t', "table")
        .required()
        .action((x, c) => c.copy(writeTable = x))
        .text("output hive table"),
      opt[String]('c', "column")
        .action((x, c) => c.copy(column = x))
        .text("output hive table column"),
      opt[Int]('d', "degree")
        .action((x, c) => c.copy(degree = x))
        .text("max degree to keep for each node"),
      opt[Double]('p', "returnParam")
        .action((x, c) => c.copy(p = x))
        .text("return parameter"),
      opt[Double]('q', "inOutParam")
        .action((x, c) => c.copy(q = x))
        .text("in-out parameter"),
      opt[Int]('n', "numWalks")
        .action((x, c) => c.copy(numWalks = x))
        .text("number of walks"),
      opt[Int]('l', "walkLength")
        .action((x, c) => c.copy(walkLength = x))
        .text("length of walks from each node"),
      opt[Int]('a', "partition")
        .action((x, c) => c.copy(numPartition = x))
        .text("number of partitions")
    )
  }

  def main(args: Array[String]): Unit = {
    logger.info(Banner.sparkLearn)
    val config = OParser.parse(parser, args, Config()) match {
      case Some(config) => {
        logger.info("hql file : "+config.hqlFile)
        logger.info("output table : "+config.writeTable)
        logger.info("max degree : "+config.degree)
        logger.info("p : "+config.p)
        logger.info("q : "+config.q)
        logger.info("number of walks : "+config.numWalks)
        logger.info("walk length : "+config.walkLength)
        logger.info("number of partitions : "+config.numPartition)
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
    val modelParams = Node2vecModel.Params(degree=config.degree, p=config.p, q=config.q,
                                           numWalks=config.numWalks, walkLength=config.walkLength, numPartition=config.numPartition)
    Node2vecModel.setup(spark, modelParams)
                 .buildGraph(df)
                 .initTransitionProb()
                 .randomWalk()
                 .saveRandomPathAsHiveTable(config.writeTable, config.column)
  }

}
