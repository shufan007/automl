package com.didi.algo.ml.automl

import com.didi.algo.ml.common._
import org.apache.spark.internal.Logging
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, Evaluator, MulticlassClassificationEvaluator}
import org.apache.spark.ml.shaded.evaluation.RegressionEvaluator
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.{CrossValidator, TrainValidationSplit}
import org.apache.spark.ml.shaded.tuning.{TrainValidationSplit => TrainValidationValidator}
import org.apache.spark.sql.DataFrame

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer


/**
 * @Author Fansx
 * @Date 2021/12/24
 * */

class AutoMLModelTuning(autoMLParams: AutoMLParams) extends Logging {
  private var estimator: Estimator[_] = _
  private var evaluator: Evaluator = _
  private var paramMaps: Array[ParamMap] = _

  def setEstimator(value: Estimator[_]): this.type = {
    estimator = value
    this
  }

  def buildEvaluator(labelCol: String,
                     taskType: String,
                     evalMetric: String="",
                     isBinary: Boolean=true): this.type = {
    //val BINARY_CLASSIFICATION_METRIC: Set[String] = Set("areaUnderROC")
    //val MULTI_CLASSIFICATION_METRIC: Set[String] = Set("f1", "accuracy", "weightedPrecision", "weightedRecall")
    //val REGRESSION_METRIC: Set[String] = Set("mae", "mape")
    val metric = evalMetric.nonEmpty match {
      case true => evalMetric
      case _ => autoMLParams.metric
    }

    evaluator = taskType match {
      case TuningParams.TASK_CLASSIFICATION => isBinary match {
        case true => new BinaryClassificationEvaluator()
          .setLabelCol(labelCol)
          .setRawPredictionCol(DEFAULT_RAW_PREDICTION_NAME)
        case false => new MulticlassClassificationEvaluator()
          .setLabelCol(labelCol)
          .setPredictionCol(DEFAULT_PREDICTION_NAME)
      }
      case TuningParams.TASK_REGRESSION => new RegressionEvaluator()
        .setLabelCol(labelCol)
        .setPredictionCol(DEFAULT_PREDICTION_NAME)
    }
    if (metric.nonEmpty) {
      taskType match {
        case TuningParams.TASK_CLASSIFICATION =>
          if (isBinary) {
            evaluator.asInstanceOf[BinaryClassificationEvaluator].setMetricName(autoMLParams.metric)
          }
          else {
            evaluator.asInstanceOf[MulticlassClassificationEvaluator].setMetricName(autoMLParams.metric)
          }
        case TuningParams.TASK_REGRESSION => {
          evaluator.asInstanceOf[RegressionEvaluator].setMetricName(autoMLParams.metric)
        }
      }
    }
    this
  }

  def paramGridBuild(turingParamsTypeMap: Map[String, String]): this.type = {
    paramMaps = new TuningParamGridBuilder(estimator.uid, autoMLParams)
      .addTuningParams(autoMLParams.tuningParams, turingParamsTypeMap)
      .build()
    //println("ParamMaps: " + paramMaps.toList)
    this
  }

  def logSearchSummary(paramMaps: Array[ParamMap], valMetrics: Array[Double]): Unit = {
    val paramsSearchSummary = ArrayBuffer[(Double, mutable.Map[String, Any])]()
    for ((paramMap, metricValue) <- (paramMaps zip valMetrics)) {
      var paramsMetricMap: mutable.Map[String, Any] = mutable.Map()
      paramMap.toSeq.foreach { paramPair =>
        paramsMetricMap.put(paramPair.param.name, paramPair.value)
      }
      paramsSearchSummary.append((metricValue, paramsMetricMap))
    }
    val sortedParamsSearchSummary = paramsSearchSummary.sortBy(x=>x._1)(
      if (evaluator.isLargerBetter) Ordering.Double.reverse else Ordering.Double)

    println("valMetrics: " + valMetrics.toList)
    println("paramsSearchSummary: " + sortedParamsSearchSummary)
  }

  def postEvaluation(model: Model[_], evaluator: Evaluator, validDF: DataFrame): Double = {
    println("Evaluation with tuning model... ")
    println("Params of best model : " + model.extractParamMap())
    try {
      val prediction = model.transform(validDF)
      println("The best model prediction... ")
      try {
        val eval = evaluator.evaluate(prediction)
        println("The best model eval is : " + eval)
        return eval
      } catch {
        case e: Exception => println("Exception: Model evaluation failed: "+ e)
        case e: Throwable => println("Unknown error: Model evaluation failed: "+ e)
      }
    } catch {
      case e: Exception => println("Exception: Model transform failed: "+ e)
      case e: Throwable => println("Unknown error: Model transform failed: "+ e)
    }
    return 0
  }

  def fit(assembleDFMap: mutable.Map[String, DataFrame]): Object = {
    /**
      Tuning model using cross validation or Train Validation Split validation
     **/

    val trainingDF = assembleDFMap(TRAIN_DATASET)

    println("start model tuning... ")
    if (assembleDFMap.contains(VALIDATE_DATASET)) {
      /**
       * user provides validation data set
       */
      println("User-provided validation data detected, will tuning with TrainValidationValidator.")

      val validationDF = assembleDFMap(VALIDATE_DATASET)

      val fittedModel = new TrainValidationValidator()
        .setEstimator(estimator)
        .setEvaluator(evaluator)
        .setEstimatorParamMaps(paramMaps)
        .setParallelism(autoMLParams.parallelism)
        .fit(trainingDF, validationDF)

      logSearchSummary(paramMaps, fittedModel.validationMetrics)
      //postEvaluation(fittedModel.bestModel, evaluator, validationDF)

      return fittedModel.bestModel
    }

    val isCv = autoMLParams.validatorType.indexOf(TuningParams.VALIDATOR_CROSS_VALIDATION) >= 0
    val fittedBestModel = isCv match {
      case true => {
        println("tuning with CrossValidator... ")
        val fittedModel = new CrossValidator()
          .setEstimator(estimator)
          .setEvaluator(evaluator)
          .setEstimatorParamMaps(paramMaps)
          .setNumFolds(autoMLParams.numFolds)
          .setParallelism(autoMLParams.parallelism)
          .fit(trainingDF)

        logSearchSummary(paramMaps, fittedModel.avgMetrics)
        //postEvaluation(fittedModel.bestModel, evaluator, trainingDF)

        fittedModel.bestModel
      }
      case false => {
        println("tuning with TrainValidationSplit... ")
        val fittedModel = new TrainValidationSplit()
          .setEstimator(estimator)
          .setEvaluator(evaluator)
          .setEstimatorParamMaps(paramMaps)
          .setTrainRatio(autoMLParams.trainRatio)
          .setParallelism(autoMLParams.parallelism)
          .fit(trainingDF)

        logSearchSummary(paramMaps, fittedModel.validationMetrics)
        //postEvaluation(fittedModel.bestModel, evaluator, trainingDF)

        fittedModel.bestModel
      }
    }
    fittedBestModel
  }
}

