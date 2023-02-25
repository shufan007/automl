package com.didi.algo.ml.automl

import breeze.stats.distributions.{Rand, Uniform}
import org.apache.spark.internal.Logging
import org.apache.spark.ml.param.{DoubleParam, IntParam, Param, ParamMap}

import scala.collection.mutable

/**
 * @Author Fansx
 * @Date 2022/01/10
 * */

class TuningParamGridBuilder(trainerUid: String, autoMLParams: AutoMLParams) extends Logging {
  private val isGrid = autoMLParams.strategyType.indexOf(TuningParams.STRATEGY_GRID_SEARCH) >= 0
  val paramGridBuilder = new ParamGroupBuilder()
  if (!isGrid) {
    paramGridBuilder.setRandomRound(autoMLParams.randomRound)
  }

  private def addParam[T](param: Param[T], paramV: Array[T]): Unit = {
    if (isGrid) {
      paramGridBuilder.addGrid(param, paramV)
    } else { // random params add
      val low = paramV(0)
      val high = paramV.length match {
        case 1 => low
        case _ => paramV(1)
      }
      low match {
        case _: Int =>
          paramGridBuilder.addDistr(param,
            Uniform(low.asInstanceOf[Int], high.asInstanceOf[Int]).map(x => scala.math.round(x).asInstanceOf[Int])
          )
        case _: Double =>
          paramGridBuilder.addDistr(param, Uniform(low.asInstanceOf[Double], high.asInstanceOf[Double]))
      }
    }
  }

  def addTuningParams(tuningParams: Map[String, Array[Any]], turingParamsTypeMap: Map[String, String]): this.type = {
    tuningParams.keys.filter(x => turingParamsTypeMap.contains(x)).foreach { paramName =>
      val paramV = tuningParams(paramName)
      if (paramV.length>0 && paramV.isInstanceOf[Array[_]]) {
        val paramType = turingParamsTypeMap(paramName)
        paramType match {
          case "Int" =>
            paramV(0) match {
              case _: Int =>
                addParam[Int](new IntParam(trainerUid, paramName, doc = ""), paramV.map(_.asInstanceOf[Int]))
              case _: BigInt =>
                addParam[Int](new IntParam(trainerUid, paramName, doc = ""), paramV.map(_.asInstanceOf[BigInt].toInt))
              case _ =>
                logError("The type of param '%s' should be '%s".format(paramName, paramType))
            }
          case "Double" =>
            paramV(0) match {
              case _: Double =>
                addParam[Double](new DoubleParam(trainerUid, paramName, doc = ""), paramV.map(_.asInstanceOf[Double]))
              case _: Int =>
                addParam[Double](new DoubleParam(trainerUid, paramName, doc = ""), paramV.map(_.asInstanceOf[Double]))
              case _: BigInt =>
                addParam[Double](new DoubleParam(trainerUid, paramName, doc = ""), paramV.map(_.asInstanceOf[BigInt].toDouble))
              case _ =>
                logError("The type of param '%s' should be '%s".format(paramName, paramType))
            }
        }
      }
    }
    this
  }

  def build(): Array[ParamMap] = {
    val paramMaps = paramGridBuilder.build()
    paramMaps
  }
}


class ParamGroupBuilder() {
  /**  randomRound: number of param groups for search, 0 as default for grid params build, positive integer for random params build
   */
  private var randomRound: Int = 0
  private val paramGrid = mutable.Map.empty[Param[_], Iterable[_]]
  private val paramDistr = mutable.Map.empty[Param[_],Any]

  def setRandomRound(value: Int): this.type = {
    this.randomRound = value
    this
  }

  def addGrid[T](param: Param[T], values: Iterable[T]): this.type = {
    paramGrid.put(param, values)
    this
  }

  def addDistr[T](param: Param[T], distr: Any ): this.type = distr match {
    case _ : Rand[_] => {paramDistr.put(param, distr)
      this}
    case _ : Array[Int] => { paramDistr.put(param, distr.asInstanceOf[Array[Int]])
      this}
    case _ : Array[Double] => { paramDistr.put(param, distr.asInstanceOf[Array[Double]])
      this}
    case _  => throw new NotImplementedError("Distribution should be of type breeze.stats.distributions.Rand or an Array")
  }

  /**
   * Builds and returns all combinations of parameters specified by the param grid.
   */
  def build(): Array[ParamMap] = {
    var paramMaps = Array(new ParamMap)
    if (this.randomRound==0) {
      paramGrid.foreach { case (param, values) =>
        val newParamMaps = values.flatMap { v =>
          paramMaps.map(_.copy.put(param.asInstanceOf[Param[Any]], v))
        }
        paramMaps = newParamMaps.toArray
      }
    } else {
      var newParamMaps = (1 to this.randomRound).map( _ => new ParamMap())
      paramDistr.foreach{
        case (param, distribution) =>
          val values = distribution match {
            case d :Rand[_] => {
              newParamMaps.map(_.put(param.asInstanceOf[Param[Any]],d.sample()))
            }
            case d : Array[Int] => {
              var r = scala.util.Random
              var _range = d.length match {
                case 1 => 1
                case _ => d(1) - d(0) + 1
              }
              newParamMaps.map(_.put(param.asInstanceOf[Param[Any]],r.nextInt(_range)+d(0)))
            }
            case d: Array[_] => {
              val r = scala.util.Random
              newParamMaps.map(_.put(param.asInstanceOf[Param[Any]], d(r.nextInt(d.length))) )
            }
          }
      }
      paramMaps = newParamMaps.toArray
    }
    paramMaps
  }
}

