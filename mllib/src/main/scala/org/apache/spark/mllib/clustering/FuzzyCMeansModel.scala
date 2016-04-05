/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.mllib.clustering

import org.apache.spark.SparkContext
import org.apache.spark.annotation.Since
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.util.Loader
import org.apache.spark.sql.{Row, SQLContext}
import org.json4s.DefaultFormats

import org.json4s._
import org.json4s.JsonDSL._
import org.json4s.jackson.JsonMethods._

/**
  * Created by acflorea on 05/04/16.
  */
class FuzzyCMeansModel(override val clusterCenters: Array[Vector])
  extends KMeansModel(clusterCenters) {

  override def save(sc: SparkContext, path: String): Unit = {
    FuzzyCMeansModel.SaveLoadV1_0.save(sc, this, path)
  }

}

@Since("1.6.0")
object FuzzyCMeansModel extends Loader[FuzzyCMeansModel] {

  override def load(sc: SparkContext, path: String): FuzzyCMeansModel = {
    FuzzyCMeansModel.SaveLoadV1_0.load(sc, path)
  }

  private case class Cluster(id: Int, point: Vector)

  private object Cluster {
    def apply(r: Row): Cluster = {
      Cluster(r.getInt(0), r.getAs[Vector](1))
    }
  }

  private[clustering]
  object SaveLoadV1_0 {

    private val thisFormatVersion = "1.0"

    private[clustering]
    val thisClassName = "org.apache.spark.mllib.clustering.FuzzyCMeansModel"

    def save(sc: SparkContext, model: FuzzyCMeansModel, path: String): Unit = {
      val sqlContext = new SQLContext(sc)
      import sqlContext.implicits._
      val metadata = compact(render(
        ("class" -> thisClassName) ~ ("version" -> thisFormatVersion) ~ ("k" -> model.k)))
      sc.parallelize(Seq(metadata), 1).saveAsTextFile(Loader.metadataPath(path))
      val dataRDD = sc.parallelize(model.clusterCenters.zipWithIndex).map { case (point, id) =>
        Cluster(id, point)
      }.toDF()
      dataRDD.write.parquet(Loader.dataPath(path))
    }

    def load(sc: SparkContext, path: String): FuzzyCMeansModel = {
      implicit val formats = DefaultFormats
      val sqlContext = new SQLContext(sc)
      val (className, formatVersion, metadata) = Loader.loadMetadata(sc, path)
      assert(className == thisClassName)
      assert(formatVersion == thisFormatVersion)
      val k = (metadata \ "k").extract[Int]
      val centroids = sqlContext.read.parquet(Loader.dataPath(path))
      Loader.checkSchema[Cluster](centroids.schema)
      val localCentroids = centroids.map(Cluster.apply).collect()
      assert(k == localCentroids.length)
      new FuzzyCMeansModel(localCentroids.sortBy(_.id).map(_.point))
    }
  }
}