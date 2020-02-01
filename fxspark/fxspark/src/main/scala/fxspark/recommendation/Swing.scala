package fxspark.recommendation

import org.slf4j.{Logger, LoggerFactory}
import scala.io.Source
import org.apache.spark.sql.{DataFrame, SparkSession}
import scopt.OParser

object Swing {
  val logger = LoggerFactory.getLogger(getClass.getName)

  case class Config(
                     hqlFile: String = "",
                     taskMode: String = "",
                     saveMode: String = "",
                     maxUser: Int = 800,
                     topn: Int = 100,
                     out: String = "",
                     saveScore: String = "",
                     overwrite: Boolean = false,
                     itemICol: String = "item_i",
                     itemJCol: String = "item_j",
                     scoreCol: String = "score",
                     itemCol: String = "item",
                     itemListCol: String = "item_list",
                     partitionCol: String = "",
                     partitionVal: String = "",
                     fieldDelimiter: String = "\t",
                     collectionDelimiter: String = ","
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
      opt[String]('m', "save-mode")
        .required()
        .action((x, c) => c.copy(saveMode = x))
        .text("save mode, hive | text"),
      opt[String]('o', "out")
        .required()
        .action((x, c) => c.copy(out = x))
        .text("output"),
      opt[Boolean]('r', "overwrite")
        .action((x, c) => c.copy(overwrite = x))
        .text("whether overwrite"),
      opt[Int]('u', "max-user")
        .action((x, c) => c.copy(maxUser = x))
        .text("max user number"),
      opt[String]("item-i")
        .action((x, c) => c.copy(itemICol = x))
        .text("item i column of item similarity hive table"),
      opt[String]("item-j")
        .action((x, c) => c.copy(itemJCol = x))
        .text("item j column of item similarity hive table"),
      opt[String]("score-col")
        .action((x, c) => c.copy(scoreCol = x))
        .text("item similarity score column of item similarity hive table"),
      opt[String]("item-col")
        .action((x, c) => c.copy(itemCol = x))
        .text("item column of topn result hive table"),
      opt[String]("itemlist-col")
        .action((x, c) => c.copy(itemListCol = x))
        .text("recommended item list column of topn result hive table"),
      opt[String]("partition-col")
        .action((x, c) => c.copy(partitionCol = x))
        .text("partition column name"),
      opt[String]("partition-val")
        .action((x, c) => c.copy(partitionVal = x))
        .text("partition column value"),
      opt[String]("field-delimiter")
        .action((x, c) => c.copy(fieldDelimiter = x))
        .text(raw"field delimiter, only used when saving in text format, default \t"),
      opt[String]("collection-delimiter")
        .action((x, c) => c.copy(collectionDelimiter = x))
        .text("collection delimiter, only used when saving in text format, default ,"),
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
        case "hive" => SwingModel.buildGraph(df, config.maxUser).computeItemSimilarity().saveItemSimilarityAsHiveTable(spark, config.out,
                        config.itemICol, config.itemJCol, config.scoreCol, config.overwrite,
                        config.partitionCol, config.partitionVal)
        case "text" => SwingModel.buildGraph(df, config.maxUser).computeItemSimilarity().saveItemSimilarityAsTextFile(spark, config.out,
          config.overwrite, config.fieldDelimiter, config.collectionDelimiter)
      }
      case "topn" => config.saveMode match {
        case "hive" => {
          if (config.saveScore.length>0) {
            SwingModel.buildGraph(df, config.maxUser).computeItemSimilarity()
              .saveItemSimilarityAsHiveTable(spark, config.saveScore, config.itemICol, config.itemJCol, config.scoreCol,
                config.overwrite, config.partitionCol, config.partitionVal)
              .computeTopn(config.topn)
              .saveTopnAsHiveTable(spark, config.out, config.itemCol, config.itemListCol, config.overwrite,
                config.partitionCol, config.partitionVal)
          }
          else {
            SwingModel.buildGraph(df, config.maxUser).computeItemSimilarity()
              .computeTopn(config.topn)
              .saveTopnAsHiveTable(spark, config.out, config.itemCol, config.itemListCol, config.overwrite,
                config.partitionCol, config.partitionVal)
          }
        }
        case "text" => {
          if (config.saveScore.length>0) {
            SwingModel.buildGraph(df, config.maxUser).computeItemSimilarity()
              .saveItemSimilarityAsTextFile(spark, config.saveScore, config.overwrite, config.fieldDelimiter, config.collectionDelimiter)
              .computeTopn(config.topn)
              .saveTopnAsTextFile(spark, config.out, config.overwrite, config.fieldDelimiter, config.collectionDelimiter)
          }
          else {
            SwingModel.buildGraph(df, config.maxUser).computeItemSimilarity()
              .computeTopn(config.topn)
              .saveTopnAsTextFile(spark, config.out, config.overwrite, config.fieldDelimiter, config.collectionDelimiter)
          }
        }
      }
    }

    spark.close()
  }
}
