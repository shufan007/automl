package com.didi.algo.ml.trainer

import com.didi.algo.ml.automl.{AutoMLParams, TuningParams, AutoMLModelTuning}
import com.didi.algo.ml.common._
import com.didi.algo.ml.util.{JsonUtils, SparkUtils}
import ml.dmlc.xgboost4j.java.RabitTracker.TrackerProcessLogger
import ml.dmlc.xgboost4j.scala.Booster
import ml.dmlc.xgboost4j.scala.spark.{XGBoostClassificationModel, XGBoostClassifier, XGBoostRegressionModel, XGBoostRegressor}
import org.apache.log4j.{ConsoleAppender, Level, Logger, PatternLayout}
import org.apache.spark.internal.Logging
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.param.{BooleanParam, Param}
import org.apache.spark.ml.shaded.{DMLParamsReader, DMLParamsWriter}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.json4s.{DefaultFormats, JArray, JString}
import org.apache.hadoop.fs.Path

import scala.math.{max, min}

/**
 * @Author Luogh
 * @Date 2021/9/26
 * */
trait XGBParams extends HasTrainParams[XGBModelParams] with HasNumClass {

  val isClassification = new BooleanParam(this, "isClassification", "isClassification")

  def getIsClassification: Boolean = $(isClassification)

  def setIsClassification(value: Boolean): this.type = set(isClassification, value)
}

@EstimatorOperator(
  name = "model_xgb_estimator",
  readerImplClazz = classOf[XGBReader],
  operatorType = OperatorTypeEnum.ModelEstimatorOperator
)
class XGBEstimator(override val uid: String) extends DMLEstimator[XGBModel] with XGBParams with Logging {

  {
    val rabitTrackerLogger = Logger.getLogger(classOf[TrackerProcessLogger])
    val appender = new ConsoleAppender(new PatternLayout(PatternLayout.TTCC_CONVERSION_PATTERN), ConsoleAppender.SYSTEM_OUT)
    appender.setThreshold(Level.INFO)
    rabitTrackerLogger.setLevel(Level.INFO)
    rabitTrackerLogger.addAppender(appender)

    val logger = Logger.getLogger("ml.dmlc.xgboost4j.java")
    logger.setLevel(Level.DEBUG)
  }

  def prepareDataFrame(df: DataFrame, features: Array[String], featuresColName: String, partitionNum: Int): DataFrame = {
    new VectorAssembler()
      .setInputCols(features)
      .setOutputCol(featuresColName)
      .setHandleInvalid("keep")
      .transform(df)
      .repartition(partitionNum)
  }

  override def fit(inputDF: Map[String, DataFrame]): DMLModel[XGBModel] = {
    require(inputDF.nonEmpty, s"input dataframe size expect at least one, but found: ${inputDF.size}")
    val trainDF = inputDF.getOrElse(TRAIN_DATASET, throw new RuntimeException("train dataset not found"))
    val numWorkers = SparkUtils.calcNumWorkers(trainDF)

    val preparedTrainDF = if ($(trainParams).weightCol.nonEmpty) {
      val weightCol = $(trainParams).weightCol
      SparkUtils.checkNumericType(trainDF.schema, weightCol)
      val weightedDF = trainDF.withColumn(weightCol, col(weightCol).cast(DoubleType))
      prepareDataFrame(weightedDF, $(trainParams).featureCols, $(trainParams).featuresCol, numWorkers)
    } else {
      prepareDataFrame(trainDF, $(trainParams).featureCols, $(trainParams).featuresCol, numWorkers)
    }

    val evalMap = if (inputDF.contains(VALIDATE_DATASET)) {
      val preparedDF = prepareDataFrame(inputDF(VALIDATE_DATASET), $(trainParams).featureCols, $(trainParams).featuresCol, numWorkers)
      Map(VALIDATE_DATASET -> preparedDF)
    } else {
      Map.empty[String, DataFrame]
    }

    val params = $(trainParams).toParamMap

    val sparkModel = if ($(isClassification)) {
      val trainer = new XGBoostClassifier()
        .setNumWorkers(numWorkers)
        .setEvalSets(evalMap)

      val isBinary = $(trainParams).objective.indexOf("binary") >= 0
      trainer.params.filter(x => params.contains(x.name)).foreach { param =>
        param.name match {
          case "weightCol" =>
            if (params(param.name) != null && params(param.name).asInstanceOf[String].nonEmpty) {
              trainer.setWeightCol(params(param.name).asInstanceOf[String])
            }
          case "evalMetric" =>
            if (params(param.name) != null && params(param.name).asInstanceOf[String].nonEmpty) {
              trainer.setEvalMetric(params(param.name).asInstanceOf[String])
            }
          case "numClass" =>
            val ignoreNumClass = "numClass".equalsIgnoreCase(param.name) && isBinary // for fix numClass bug for binary
            if (!ignoreNumClass) {
              trainer.setNumClass(params(param.name).asInstanceOf[Int])
            }
          case _ =>
            val paramV = params(param.name)
            trainer.set(param.asInstanceOf[Param[Any]], paramV)
        }
      }

      val fittedModel = $(trainParams).autoMLParams match {
        case Some(autoMLParams) =>
          println("autoMLParams: " + JsonUtils.toJsonString(autoMLParams))
          val assembleDFMap = scala.collection.mutable.Map(TRAIN_DATASET->preparedTrainDF)
          if (evalMap.nonEmpty) {
            assembleDFMap(VALIDATE_DATASET) = evalMap(VALIDATE_DATASET)
          }
          val bestModel = new AutoMLModelTuning(autoMLParams)
            .setEstimator(trainer.setEvalSets(Map.empty[String, DataFrame]))
            .buildEvaluator($(trainParams).labelCol, TuningParams.TASK_CLASSIFICATION, $(trainParams).evalMetric, isBinary)
            .paramGridBuild(TuningParams.XGB_TUNING_PARAMS_TYPE_MAP)
            .fit(assembleDFMap)
            .asInstanceOf[XGBoostClassificationModel]
          bestModel
        case None =>
          trainer.fit(preparedTrainDF)
      }
      Left(fittedModel)
    } else {
      val trainer = new XGBoostRegressor()
        .setNumWorkers(numWorkers)
        .setEvalSets(evalMap)

      trainer.params.filter(x => params.contains(x.name)).foreach { param =>
        param.name match {
          case "weightCol" =>
            if (params(param.name) != null && params(param.name).asInstanceOf[String].nonEmpty) {
              trainer.setWeightCol(params(param.name).asInstanceOf[String])
            }
          case "evalMetric" =>
            if (params(param.name) != null && params(param.name).asInstanceOf[String].nonEmpty) {
              trainer.setEvalMetric(params(param.name).asInstanceOf[String])
            }
          case _ =>
            val paramV = params(param.name)
            trainer.set(param.asInstanceOf[Param[Any]], paramV)
        }
      }

      val fittedModel = $(trainParams).autoMLParams match {
        case Some(autoMLParams) =>
          println("autoMLParams: " + JsonUtils.toJsonString(autoMLParams))
          val assembleDFMap = scala.collection.mutable.Map(TRAIN_DATASET->preparedTrainDF)
          if (evalMap.nonEmpty) {
            assembleDFMap(VALIDATE_DATASET) = evalMap(VALIDATE_DATASET)
          }
          val bestModel = new AutoMLModelTuning(autoMLParams)
            .setEstimator(trainer.setEvalSets(Map.empty[String, DataFrame]))
            .buildEvaluator($(trainParams).labelCol, TuningParams.TASK_REGRESSION, $(trainParams).evalMetric)
            .paramGridBuild(TuningParams.XGB_TUNING_PARAMS_TYPE_MAP)
            .fit(assembleDFMap)
            .asInstanceOf[XGBoostRegressionModel]
          bestModel
        case None =>
          trainer.fit(preparedTrainDF)
      }

      Right(fittedModel)
    }
    val xGBModel = new XGBModel(this.uid, XGBModelState($(isClassification), $(trainParams).labelCol,
      $(trainParams).predictionCol, $(trainParams).featuresCol, $(trainParams).featureCols), sparkModel)
    copyValues(xGBModel)
  }
}

object XGBEstimator extends DMLModelEstimatorBuilder[XGBModel] {

  val CLASSIFICATION_OBJECTIVE: Set[String] = Set("binary:logistic", "multi:softmax", "multi:softprob")
  val REGRESSION_OBJECTIVE: Set[String] = Set("reg:squarederror", "reg:logistic", "reg:gamma", "reg:tweedie")

  override def build(trainerParams: ModelEstimatorParams): DMLEstimator[XGBModel] = {
    require(trainerParams.modelPath.nonEmpty, "模型保存路径未指定")
    require(trainerParams.modelParams.nonEmpty, "模型超参未指定")
    require(trainerParams.tableName.nonEmpty
      || trainerParams.tableNames.keySet.forall(x => x.equalsIgnoreCase(TRAIN_DATASET) || x.equalsIgnoreCase(VALIDATE_DATASET)),
      "输入数据集未指定或无效数据集")
    val modelParams = JsonUtils.parseJson[XGBModelParams](trainerParams.modelParams.get)
    require(modelParams.labelCol.nonEmpty, "labelCol must specified")
    require(modelParams.objective.nonEmpty && (CLASSIFICATION_OBJECTIVE.contains(modelParams.objective) || REGRESSION_OBJECTIVE.contains(modelParams.objective)), "objective must specified")

    val estimator = new XGBEstimator(Identifiable.randomUID("xgbEstimator"))
      .setIsClassification(CLASSIFICATION_OBJECTIVE.contains(modelParams.objective))
      .setTrainParams(modelParams)

    estimator
  }
}


class XGBModel(
                override val uid: String,
                override val state: XGBModelState,
                val sparkModel: Either[XGBoostClassificationModel, XGBoostRegressionModel]
              ) extends DMLModel[XGBModel](uid, state) {
  override def transform(inputDF: Map[String, DataFrame]): DataFrame = {
    require(inputDF.size == 1, s"input dataframe size expect only one, but found: ${inputDF.size}")
    val dataset = inputDF.head._2
    val assembleDF = new VectorAssembler()
      .setInputCols(state.features)
      .setOutputCol(state.featuresCol)
      .setHandleInvalid("keep")
      .transform(dataset)
    if (state.isClassification) {
      sparkModel.left.getOrElse(throw new RuntimeException("classification model not found"))
        .transform(assembleDF)
    } else {
      sparkModel.right.getOrElse(throw new RuntimeException("regression model not found"))
        .transform(assembleDF)
    }
  }

  override def write: DMLWriter = new XGBWriter(this)
}

class XGBReader extends DMLReader[XGBModel] {
  override def load(path: String): XGBModel = {
    val metadata = DMLParamsReader.loadMetadata(path, sc, classOf[XGBModel].getName)
    implicit val format: DefaultFormats.type = DefaultFormats
    val metaParams = metadata.metadata
    val isClassification = (metaParams \ META_IS_CLASSIFICATION).extract[Boolean]
    val labelColName = (metaParams \ META_LABEL_COL).extract[String]
    val predictionColName = (metaParams \ META_PREDICTION_COL).extract[String]
    val featuresCol = (metaParams \ META_FEATURES_COL).extract[String]
    val features = (metaParams \ META_FEATURES).extract[Array[String]]
    val modelState = XGBModelState(isClassification, labelColName, predictionColName, featuresCol, features)
    // load model
    val sparkModel = if (isClassification) {
      Left(XGBoostClassificationModel.load(modelDirPath(path)))
    } else {
      Right(XGBoostRegressionModel.load(modelDirPath(path)))
    }
    new XGBModel(metadata.uid, modelState, sparkModel)
  }
}

class XGBWriter(override val model: XGBModel) extends DMLWriter(model) {
  override protected def saveImpl(path: String): Unit = {
    import org.json4s.JsonDSL._
    var numClass: Int = -1
    val modelPath = new Path(modelDirPath(path), "data").toString
    model.sparkModel match {
      case Left(sparkModel) =>
        numClass = sparkModel.numClasses
        val nativeModelPath = new Path(modelPath, "XGBoostClassificationModel").toString
        saveNativeModel(sparkSession, sparkModel.nativeBooster, nativeModelPath)
        saveFeatureImportance(sparkSession, sparkModel.nativeBooster, model.state.features, featureImportanceFilePath(path))
      case Right(sparkModel) =>
        val nativeModelPath = new Path(modelPath, "XGBoostRegressionModel").toString
        saveNativeModel(sparkSession, sparkModel.nativeBooster, nativeModelPath)
        saveFeatureImportance(sparkSession, sparkModel.nativeBooster, model.state.features, featureImportanceFilePath(path))
    }
    val extraMetaParams =
      (META_MODEL_READER_IMPL_CLAZZ -> classOf[XGBReader].getName) ~
        (META_IS_CLASSIFICATION -> model.state.isClassification) ~
        (META_NUM_CLASS -> numClass) ~
        (META_LABEL_COL -> model.state.labelColName) ~
        (META_PREDICTION_COL -> model.state.predictionColName) ~
        (META_FEATURES_COL -> model.state.featuresCol) ~
        (META_FEATURES -> JArray(model.state.features.map(JString).toList))
    DMLParamsWriter.saveMetadata(model, path, sc, Some(extraMetaParams))
  }

  private def saveNativeModel(sc: SparkSession, booster: Booster, path: String): Unit = {
    val internalPath = new Path(path)
    val outputStream = internalPath.getFileSystem(sc.sparkContext.hadoopConfiguration).create(internalPath)
    booster.saveModel(outputStream)
    outputStream.close()
  }

  private def saveFeatureImportance(sc: SparkSession, model: Booster, features: Array[String], path: String): Unit = {
    val featureImportance = model.getScore(features, XGB_DEFAULT_FEATURE_IMPORTANCE_TYPE)
      .toSeq.map {
      case (k, i) => FeatureImportance(k, i)
    }.sortBy(_.importance).reverse
    SparkUtils.serializeDataToHDFS(sc.sparkContext, JsonUtils.toJsonString(featureImportance), path, overwrite = true)
  }
}


case class XGBModelState(isClassification: Boolean,
                         labelColName: String,
                         predictionColName: String,
                         featuresCol: String,
                         features: Array[String] = Array.empty[String]) extends ModelParamState

case class XGBModelParams(
                           featureCols: Array[String] = Array.empty[String],
                           weightCol: String = "",
                           featuresCol: String = DEFAULT_FEATURES_NAME,
                           labelCol: String = DEFAULT_LABEL_NAME,
                           predictionCol: String = DEFAULT_PREDICTION_NAME,
                           objective: String = "",
                           numEarlyStoppingRounds: Int = 0,
                           eta: Double = 0.3,
                           maxDepth: Int = 6,
                           subsample: Double = 1,
                           lambda: Double = 1,
                           alpha: Double = 0,
                           minChildWeight: Double = 1,
                           seed: Long = 0,
                           numRound: Int = 1,
                           gamma: Double = 0,
                           sketchEps: Double = 0.03,
                           colSampleByTree: Double = 1,
                           missing: Float = Float.NaN,
                           numClass: Int = 2,
                           tweedieVariancePower: Double = 1.5,
                           scalePosWeight: Double = 1.0,
                           evalMetric: String = "",
                           autoMLParams: Option[AutoMLParams] = None
                         ) extends AbstractParams[XGBModelParams]


