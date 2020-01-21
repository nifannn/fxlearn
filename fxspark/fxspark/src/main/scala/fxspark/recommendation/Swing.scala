package fxspark.recommendation

import org.slf4j.{Logger, LoggerFactory}
import scala.io.Source
import org.apache.spark.sql.{DataFrame, SparkSession}
import scopt.OParser
import fxspark.recommendation.SwingModel

object Swing {
  val logger = LoggerFactory.getLogger(getClass.getName)

  case class Config(
                     hqlFile: String = "",
                     taskMode: String = "",
                     saveMode: String = "",
                     topn: Int = 100,
                     out: String = "",
                     saveScore: String = ""
                   )
  val builder = OParser.builder[Config]
  val parser = {
    import builder._
    OParser.sequence(
      programName("spark-submit --class fxspark.recommendation.Swing "),
      head("fxspark", "1.0"),
      opt[String]('f', "hql")
        .required()
        .valueName("<hql_file>")
        .action((x, c) => c.copy(hqlFile = x))
        .text("input hql file"),
      opt[String]('s', "save-mode")
        .required()
        .action((x, c) => c.copy(saveMode = x))
        .text("save mode, hive | text"),
      opt[String]('o', "out")
        .required()
        .action((x, c) => c.copy(out = x))
        .text("output"),
      cmd("score")
        .action((_, c) => c.copy(taskMode = "score"))
        .text("compute item similarity only"),
      cmd("topn")
        .action((_, c) => c.copy(taskMode = "topn"))
        .text("recommend topn items")
        .children(
          opt[Int]("n")
            .action((x, c) => c.copy(topn = x))
            .text("number of items to recommend"),
          opt[String]("save-score")
            .action((x, c) => c.copy(saveScore = x))
            .text("path to save item similarity, which will not be saved if not set ")
        )
    )
  }

  def main(args: Array[String]): Unit = {
    val config = OParser.parse(parser, args, Config()) match {
      case Some(config) => {
        logger.info("hql file : "+config.hqlFile)
        logger.info("task : "+config.taskMode)
        logger.info("save mode : "+config.saveMode)
        logger.info("output : "+config.out)
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

    val spark = SparkSession
      .builder()
      .appName(s"${this.getClass.getSimpleName}")
      .enableHiveSupport()
      .getOrCreate()

    val df = spark.sql(hql)

    config.taskMode match {
      case "score" => config.saveMode match {
        case "hive" => SwingModel.computeItemSimilarity(df).saveItemSimilarityAsHiveTable(config.out)
        case "text" => SwingModel.computeItemSimilarity(df).saveItemSimilarityAsTextFile(config.out)
      }
      case "topn" => config.saveMode match {
        case "hive" => {
          if config.saveScore.length > 0 SwingModel.computeItemSimilarity
        }
      }
    }

    spark.close()
  }
}
