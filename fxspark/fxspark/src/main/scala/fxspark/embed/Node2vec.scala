package fxspark.embed

import org.slf4j.{Logger, LoggerFactory}
import scala.io.Source
import org.apache.spark.sql.{DataFrame, SparkSession}
import scopt.OParser

object Node2vec {
  val logger = LoggerFactory.getLogger(getClass.getName)

  case class Config(
                   hqlFile: String = "",
                   taskMode: String = "",
                   saveMode: String = "",
                   out: String = "",
                   savePath: String = "",
                   overwrite: Boolean = true,
                   pathCol: String = "path",
                   nodeCol: String = "node",
                   embedCol: String = "embedding",
                   partitionCol: String = "",
                   partitionVal: String = "",
                   fieldDelimiter: String = "\t",
                   pathDelimiter: String = " ",
                   embedDelimiter: String = " ",
                   degree: Int = 30,
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
                   stepSize: Double = 0.025,
                   numPartition: Int = 64)

  val builder = OParser.builder[Config]
  val parser = {
    import builder._
    OParser.sequence(
      programName("spark-submit --class fxspark.embed.Node2vec "),
      head("fxspark", "1.0"),
      opt[String]('f', "hql")
        .required()
        .valueName("<hql_file>")
        .action((x, c) => c.copy(hqlFile = x))
        .text("hql file"),
      opt[String]('o', "out")
        .required()
        .action((x, c) => c.copy(out = x))
        .text("output table or path"),
      opt[String]('m', "save-mode")
        .required()
        .action((x, c) => c.copy(saveMode = x))
        .text("save mode, hive | text"),
      opt[Boolean]("overwrite")
        .action((x, c) => c.copy(overwrite = x))
        .text("whether overwrite"),
      opt[Int]('d', "degree")
        .action((x, c) => c.copy(degree = x))
        .text("max degree to keep for each node"),
      opt[Double]('p', "return-param")
        .action((x, c) => c.copy(p = x))
        .text("return parameter"),
      opt[Double]('q', "inout-param")
        .action((x, c) => c.copy(q = x))
        .text("inout parameter"),
      opt[Int]('n', "num-walks")
        .action((x, c) => c.copy(numWalks = x))
        .text("number of walks"),
      opt[Int]('l', "walk-length")
        .action((x, c) => c.copy(walkLength = x))
        .text("length of walks from each node"),
      opt[Boolean]("directed")
        .action((x, c) => c.copy(directed = x))
        .text("whether directed graph"),
      opt[Boolean]("weighted")
        .action((x, c) => c.copy(weighted = x))
        .text("whether weighted graph"),
      opt[String]("path-col")
        .action((x, c) => c.copy(pathCol = x))
        .text("path column of random walk paths hive table"),
      opt[String]("path-delimiter")
        .action((x, c) => c.copy(pathDelimiter = x))
        .text("path delimiter, used when saving random walk paths in text format"),
      opt[String]("partition-col")
        .action((x, c) => c.copy(partitionCol = x))
        .text("partition column, for hive table"),
      opt[String]("partition-val")
        .action((x, c) => c.copy(partitionVal = x))
        .text("partition value, for hive table"),
      cmd("random-walk")
        .action((_, c) => c.copy(taskMode = "random-walk"))
        .text("random walk only"),
      cmd("embedding")
        .action((_, c) => c.copy(taskMode = "embedding"))
        .text("random walk and train embedding")
        .children(
          opt[String]("save-path")
            .action((x, c) => c.copy(savePath = x))
            .text("path or table to save random walk paths, if not set, paths will not be saved"),
          opt[String]("node-col")
            .action((x, c) => c.copy(nodeCol = x))
            .text("node column of embedding hive table"),
          opt[String]("embed-col")
            .action((x, c) => c.copy(embedCol = x))
            .text("embedding column of embedding hive table"),
          opt[String]("field-delimiter")
            .action((x, c) => c.copy(fieldDelimiter = x))
            .text("field delimiter, used when saving embeddings in text format"),
          opt[String]("embed-delimiter")
            .action((x, c) => c.copy(embedDelimiter = x))
            .text("embedding delimiter, used when saving embeddings in text format"),
          opt[Int]("embed-dim")
            .action((x, c) => c.copy(embedDim = x))
            .text("embedding dimension"),
          opt[Int]("max-iter")
            .action((x, c) => c.copy(maxIter = x))
            .text("number of iterations for training embeddings"),
          opt[Int]("min-count")
            .action((x, c) => c.copy(minCount = x))
            .text("minimum number of times a token must appear to be included in the word2vec model's vocabulary"),
          opt[Int]("window-size")
            .action((x, c) => c.copy(windowSize = x))
            .text("window size"),
          opt[Double]("step-size")
            .action((x, c) => c.copy(stepSize = x))
            .text("step size used for each iteration of optimization"),
          opt[Int]("num-partition")
            .action((x, c) => c.copy(numPartition = x))
            .text("number of partitions when training embedding")
        )
    )
  }

  def main(args: Array[String]): Unit = {
    val config = OParser.parse(parser, args, Config()) match {
      case Some(config) => {
        logger.info("hql file : "+config.hqlFile)
        logger.info("output : "+config.out)
        logger.info("max degree : "+config.degree)
        logger.info("p : "+config.p)
        logger.info("q : "+config.q)
        logger.info("number of walks : "+config.numWalks)
        logger.info("walk length : "+config.walkLength)
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

    val spark = SparkSession.builder()
      .appName(s"${this.getClass.getSimpleName}")
      .enableHiveSupport()
      .getOrCreate()

    val df = spark.sql(hql)
    config.taskMode match {
      case "random-walk" => config.saveMode match {
        case "hive" => Node2vecModel.setParams(config.degree, config.p, config.q, config.numWalks, config.walkLength,
          config.directed, config.weighted).buildGraph(spark, df).initTransitionProb(spark).randomWalkAndSavePathAsHiveTable(spark,
          config.out, config.overwrite, config.pathDelimiter, config.pathCol, config.partitionCol, config.partitionVal)

        case "text" => Node2vecModel.setParams(config.degree, config.p, config.q, config.numWalks, config.walkLength,
          config.directed, config.weighted).buildGraph(spark, df).initTransitionProb(spark).randomWalkAndSavePathAsTextFile(spark,
          config.out, config.overwrite, config.pathDelimiter)
      }
      case "embedding" => config.saveMode match {
        case "hive" => {
          if (config.savePath.length > 0){
            Node2vecModel.setParams(config.degree, config.p, config.q, config.numWalks, config.walkLength,
              config.directed, config.weighted, config.embedDim, config.maxIter, config.minCount, config.windowSize,
              config.stepSize).buildGraph(spark, df).initTransitionProb(spark).randomWalk().saveRandomPathAsHiveTable(
              spark, config.savePath, config.overwrite, config.pathDelimiter, config.pathCol,
              config.partitionCol, config.partitionVal).trainEmbedding(spark, config.numPartition).saveEmbeddingAsHiveTable(spark, config.out,
              config.overwrite, config.embedDelimiter, config.nodeCol, config.embedCol, config.partitionCol, config.partitionVal)
          } else {
            Node2vecModel.setParams(config.degree, config.p, config.q, config.numWalks, config.walkLength,
              config.directed, config.weighted, config.embedDim, config.maxIter, config.minCount, config.windowSize,
              config.stepSize).buildGraph(spark, df).initTransitionProb(spark).randomWalk().trainEmbedding(
              spark, config.numPartition).saveEmbeddingAsHiveTable(spark, config.out, config.overwrite, config.embedDelimiter,
              config.nodeCol, config.embedCol, config.partitionCol, config.partitionVal)
          }
        }
        case "text" => {
          if (config.savePath.length > 0){
            Node2vecModel.setParams(config.degree, config.p, config.q, config.numWalks, config.walkLength,
              config.directed, config.weighted, config.embedDim, config.maxIter, config.minCount, config.windowSize,
              config.stepSize).buildGraph(spark, df).initTransitionProb(spark).randomWalk().saveRandomPathAsTextFile(
              spark, config.savePath, config.overwrite, config.pathDelimiter).trainEmbedding(spark, config.numPartition).saveEmbeddingAsTextFile(spark, config.out,
              config.overwrite, config.fieldDelimiter, config.embedDelimiter)
          } else {
            Node2vecModel.setParams(config.degree, config.p, config.q, config.numWalks, config.walkLength,
              config.directed, config.weighted, config.embedDim, config.maxIter, config.minCount, config.windowSize,
              config.stepSize).buildGraph(spark, df).initTransitionProb(spark).randomWalk().trainEmbedding(
              spark, config.numPartition).saveEmbeddingAsTextFile(spark, config.out, config.overwrite, config.fieldDelimiter, config.embedDelimiter)
          }
        }
      }
    }
  }
}
