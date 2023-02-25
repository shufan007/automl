package com.didi.algo.ml.automl

/**
 * @Author Fansx
 * @Date 2022/01/13
 * */

object TuningParams {

  val VALIDATOR_CROSS_VALIDATION: String = "CROSS_VALIDATION"
  val VALIDATOR_TRAIN_VALID_SPLIT: String = "TRAIN_VALID_SPLIT"

  val STRATEGY_GRID_SEARCH: String = "GRID_SEARCH"
  val STRATEGY_RANDOM_SEARCH: String = "RANDOM_SEARCH"

  val TASK_CLASSIFICATION: String = "CLASSIFICATION"
  val TASK_REGRESSION: String = "REGRESSION"
  val TASK_RANKING: String = "RANKING"

  val XGB_TUNING_PARAMS_TYPE_MAP: Map[String, String] = Map(
    "eta" -> "Double",
    "maxDepth" -> "Int",
    "numRound" -> "Int",
    "maxBin" -> "Int",
    "minChildWeight" -> "Double",
    "lambda" -> "Double",
    "alpha" -> "Double"
  )

  val GBM_TUNING_PARAMS_TYPE_MAP: Map[String, String] = Map(
    "learningRate" -> "Double",
    "maxDepth" -> "Int",
    "numLeaves" -> "Int",
    "numIterations" -> "Int",
    "maxBin" -> "Int",
    "minSumHessianInLeaf" -> "Double",
    "lambdaL1" -> "Double",
    "lambdaL2" -> "Double"
  )

}

case class AutoMLParams(
                         metric: String = "", //ModelEvaluateMetricEnum
                         validatorType: String = TuningParams.VALIDATOR_TRAIN_VALID_SPLIT,
                         trainRatio: Double = 0.75,
                         numFolds: Int = 3, // number of cross_validator folds
                         strategyType: String = TuningParams.STRATEGY_RANDOM_SEARCH,
                         randomRound: Int = 1,  // number of random search round
                         parallelism: Int = 1,
                         tuningParams: Map[String, Array[Any]]
                       )





