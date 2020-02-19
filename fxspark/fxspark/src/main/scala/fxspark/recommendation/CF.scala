package fxspark.recommendation

import org.slf4j.{Logger, LoggerFactory}
import scala.io.Source
import org.apache.spark.sql.{DataFrame, SparkSession}
import scopt.OParser

object CF {
  val logger = LoggerFactory.getLogger(getClass.getName)

  case class Config(
                     hqlFile: String = "",
                     taskMode: String = "",
                     saveMode: String = "",
                     modelType: String = "",
                     similarFormula: String = "",
                     topn: Int = 20,
                     out: String = "",
                     saveScore: String = "",
                     overwrite: Boolean = false,
                     colI: String = "i",
                     colJ: String = "j",
                     scoreCol: String = "score",
                     mainCol: String = "id",
                     recCol: String = "rec_list",
                     partitionCol: String = "",
                     partitionVal: String = "",
                     fieldDelimiter: String = "\t",
                     collectionDelimiter: String = ","
                   )

  val builder = OParser.builder[Config]
  val parser = {
    import builder._
    OParser.sequence(
      programName("spark-submit --class fxspark.recommendation.CF "),
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
      opt[String]('t', "model-type")
        .required()
        .action((x, c) => c.copy(modelType = x))
        .text("model type, user | item"),
      opt[String]('o', "out")
        .required()
        .action((x, c) => c.copy(out = x))
        .text("output"),
      opt[String]("formula")
        .required()
        .action((x, c) => c.copy(similarFormula = x))
        .text("similarity formula"),
      opt[Boolean]('r', "overwrite")
        .action((x, c) => c.copy(overwrite = x))
        .text("whether overwrite"),
      opt[String]("col-i")
        .action((x, c) => c.copy(colI = x))
        .text("column i of similarity hive table"),
      opt[String]("col-j")
        .action((x, c) => c.copy(colJ = x))
        .text("column j of similarity hive table"),
      opt[String]("score-col")
        .action((x, c) => c.copy(scoreCol = x))
        .text("similarity score column of similarity hive table"),
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
        .text("compute similarity only"),
      cmd("topn")
        .action((_, c) => c.copy(taskMode = "topn"))
        .text("recommend topn")
        .children(
          opt[Int]("n")
            .action((x, c) => c.copy(topn = x))
            .text("topn to recommend"),
          opt[String]("save-score")
            .action((x, c) => c.copy(saveScore = x))
            .text("path to save similarity, which will not be saved if not set "),
          opt[String]("main-col")
            .action((x, c) => c.copy(mainCol = x))
            .text("main column of topn result hive table"),
          opt[String]("rec-col")
            .action((x, c) => c.copy(recCol = x))
            .text("recommended list column of topn result hive table")
        )
    )
  }

  def main(args: Array[String]): Unit = {
    val config = OParser.parse(parser, args, Config()) match {
      case Some(config) => {
        logger.info("hql file : "+config.hqlFile)
        logger.info("model : "+config.modelType + " cf")
        logger.info("formula : "+config.similarFormula)
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
        case "hive" => CFModel.computeSimilarity(spark, df, config.modelType, config.similarFormula)
            .saveSimilarityAsHiveTable(config.out,  config.overwrite, config.colI, config.colJ, config.scoreCol,
              config.partitionCol, config.partitionVal)
        case "text" => CFModel.computeSimilarity(spark, df, config.modelType, config.similarFormula)
            .saveSimilarityAsTextFile(spark, config.out, config.overwrite, config.fieldDelimiter)
      }
      case "topn" => config.saveMode match {
        case "hive" => {
          if (config.saveScore.length > 0) {
            CFModel.computeSimilarity(spark, df, config.modelType, config.similarFormula)
              .saveSimilarityAsHiveTable(config.saveScore,  config.overwrite, config.colI, config.colJ, config.scoreCol,
                config.partitionCol, config.partitionVal).computeTopn(spark, config.topn).saveTopnAsHiveTable(config.out,
              config.overwrite, config.mainCol, config.recCol, config.partitionCol, config.partitionVal)
          } else {
            CFModel.computeSimilarity(spark, df, config.modelType, config.similarFormula).computeTopn(spark,
              config.topn).saveTopnAsHiveTable(config.out, config.overwrite, config.mainCol, config.recCol,
              config.partitionCol, config.partitionVal)
          }
        }
        case "text" =>{
          if (config.saveScore.length >0) {
            CFModel.computeSimilarity(spark, df, config.modelType, config.similarFormula)
              .saveSimilarityAsTextFile(spark, config.saveScore,  config.overwrite, config.fieldDelimiter).computeTopn(spark,
              config.topn).saveTopnAsTextFile(spark, config.out, config.overwrite, config.fieldDelimiter, config.collectionDelimiter)
          } else {
            CFModel.computeSimilarity(spark, df, config.modelType, config.similarFormula).computeTopn(spark,
              config.topn).saveTopnAsTextFile(spark, config.out, config.overwrite, config.fieldDelimiter, config.collectionDelimiter)
          }
        }
      }
    }
  }
}